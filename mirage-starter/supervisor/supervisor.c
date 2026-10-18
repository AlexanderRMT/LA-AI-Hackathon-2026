/*
 * supervisor.c: Mirage safety supervisor for QNX 8.0            owner: Alex
 *
 * The only process that touches the relay. The gas valve stays open only
 * while fresh "clear" verdicts keep arriving from the vision process.
 *   - MIRAGE_TRIP_RUN gas verdicts in a row   -> trip, valve closes, latches
 *   - no verdict for MIRAGE_DEADLINE_MS       -> trip (fail closed)
 *   - boots tripped; a person reopens it, and only after clean verdicts
 *
 * Build for the Pi:  qcc -Vgcc_ntoaarch64le -O2 -o supervisor supervisor.c -lsocket
 * Build for the VM:  qcc -Vgcc_ntox86_64 -O2 -DNO_GPIO -o supervisor supervisor.c -lsocket
 * Run as root:       ./supervisor [dashboard_ip] [port]      (port defaults to 5005)
 */
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/dispatch.h>
#include <sys/neutrino.h>
#ifndef NO_GPIO
#include <sys/rpi_gpio.h>   /* if missing, copy it from the QNX hardware samples repo (common/) */
#endif
#include "../common/mirage_msgs.h"

#define SUP_PRIORITY    50
#define TELEM_PRIORITY  10
#define TICK_MS         10
#define PULSE_TICK      (_PULSE_CODE_MINAVAIL + 1)
#define BUZZ_MS         1500

/* BCM GPIO numbers, physical header pin in parentheses */
#define PIN_RELAY       17      /* (11) relay IN, energized = valve open */
#define PIN_BUZZER      23      /* (16) active buzzer */
/* LED panel, row 1: gas and valve */
#define PIN_LED_OPEN    21      /* (40) green: valve open */
#define PIN_LED_LEAK    20      /* (38) red: tripped on a leak */
#define PIN_LED_GAS     16      /* (36) yellow: the latest verdict is gas */
/* LED panel, row 2: system health */
#define PIN_LED_ALIVE   13      /* (33) green: blinks while vision's verdicts arrive in time */
#define PIN_LED_READY   19      /* (35) yellow: tripped, and a reset would be accepted now */
#define PIN_LED_FAULT   26      /* (37) red: tripped because vision went silent, or still booting */
#define ALIVE_BLINK_VERDICTS 3  /* alive LED toggles every this many verdicts (~1.7 Hz at 10 Hz) */
#define PIN_BUTTON      24      /* (18) reset button */
#define RELAY_ACTIVE_HIGH    1   /* 0 if your relay module switches on with a LOW input */
#define BUTTON_PRESSED_LEVEL 0   /* pin is pulled up, so pressed reads 0; flip if your module drives high */

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* ------------------------------------------------------------------ GPIO */
/* Uses the rpi_gpio resource manager: messages to /dev/gpio/msg for the outputs
 * and the button. Check `ls /dev/gpio` on the Pi. */

#ifndef NO_GPIO
static int gpio_fd = -1;

static int gpio_msg(unsigned subtype, unsigned gpio, unsigned value)
{
    rpi_gpio_msg_t msg;
    memset(&msg, 0, sizeof msg);
    msg.hdr.type = _IO_MSG;
    msg.hdr.subtype = subtype;
    msg.hdr.mgrid = RPI_GPIO_IOMGR;
    msg.gpio = gpio;
    msg.value = value;
    return MsgSend(gpio_fd, &msg, sizeof msg, NULL, 0);
}
#endif

static void gpio_setup(void)
{
#ifndef NO_GPIO
    gpio_fd = open("/dev/gpio/msg", O_RDWR);
    if (gpio_fd == -1) {
        perror("[supervisor] open /dev/gpio/msg, continuing without GPIO");
        return;
    }
    const unsigned outs[] = { PIN_RELAY, PIN_BUZZER, PIN_LED_OPEN, PIN_LED_LEAK, PIN_LED_GAS,
                              PIN_LED_ALIVE, PIN_LED_READY, PIN_LED_FAULT };
    for (unsigned i = 0; i < sizeof outs / sizeof outs[0]; i++) {
        if (gpio_msg(RPI_GPIO_SET_SELECT, outs[i], RPI_GPIO_FUNC_OUT) == -1) {
            perror("[supervisor] set GPIO as output");
        }
    }
    /* Pull-up so an unwired or loose button reads as released, never as a press */
    if (gpio_msg(RPI_GPIO_SET_SELECT, PIN_BUTTON, RPI_GPIO_FUNC_IN) == -1 ||
        gpio_msg(RPI_GPIO_PUD, PIN_BUTTON, RPI_GPIO_PUD_UP) == -1) {
        perror("[supervisor] set button GPIO as input");
    }
#else
    fprintf(stderr, "[supervisor] built with NO_GPIO, outputs are printed only\n");
#endif
}

/* Writes only when a level changes, so the 100 Hz tick costs nothing. */
static int cached_level[32];
static void gpio_set(unsigned gpio, int level)
{
    level = level ? 1 : 0;
    if (gpio < 32 && cached_level[gpio] == level + 1) {
        return;
    }
    if (gpio < 32) {
        cached_level[gpio] = level + 1;
    }
#ifndef NO_GPIO
    if (gpio_fd != -1 && gpio_msg(RPI_GPIO_WRITE, gpio, (unsigned)level) == -1) {
        perror("[supervisor] GPIO write");
    }
#else
    fprintf(stderr, "[gpio] %u -> %d\n", gpio, level);
#endif
}

static int button_down(void)
{
#ifndef NO_GPIO
    if (gpio_fd == -1) {
        return 0;
    }
    rpi_gpio_msg_t msg;
    memset(&msg, 0, sizeof msg);
    msg.hdr.type = _IO_MSG;
    msg.hdr.subtype = RPI_GPIO_READ;
    msg.hdr.mgrid = RPI_GPIO_IOMGR;
    msg.gpio = PIN_BUTTON;
    if (MsgSend(gpio_fd, &msg, sizeof msg, &msg, sizeof msg) == -1) {
        return 0;
    }
    return msg.value == BUTTON_PRESSED_LEVEL;
#else
    return 0;
#endif
}

/* ----------------------------------------------------------------- state */

typedef struct {
    pthread_mutex_t lock;
    int      tripped;
    uint32_t reason;
    uint64_t trip_ns;
    uint64_t last_verdict_ns;
    uint32_t last_seq;
    uint8_t  last_cls;
    float    last_p_gas;
    float    last_peak;
    int      gas_run;
    int      clear_run;
    uint64_t gas_start_frame_ns;
    float    detect_to_trip_ms;   /* first gas frame -> valve command */
    float    lat_ms;              /* frame timestamp -> verdict received */
    float    lat_max_ms;
    uint32_t n_verdicts;
    volatile int reset_request;
} sup_t;

static sup_t S = { .lock = PTHREAD_MUTEX_INITIALIZER, .tripped = 1, .reason = MIRAGE_TRIP_BOOT };

static const char *reason_str(uint32_t r)
{
    switch (r) {
    case MIRAGE_TRIP_BOOT:     return "boot";
    case MIRAGE_TRIP_LEAK:     return "leak";
    case MIRAGE_TRIP_DEADLINE: return "deadline";
    default:                   return "none";
    }
}

static void apply_outputs_locked(uint64_t t)
{
    const int open = !S.tripped;
    const int fresh = S.last_verdict_ns &&
                      (t - S.last_verdict_ns) < (uint64_t)MIRAGE_DEADLINE_MS * 1000000ull;
    gpio_set(PIN_RELAY, RELAY_ACTIVE_HIGH ? open : !open);
    gpio_set(PIN_BUZZER, S.tripped && S.reason != MIRAGE_TRIP_BOOT &&
                         (t - S.trip_ns) < (uint64_t)BUZZ_MS * 1000000ull);
    gpio_set(PIN_LED_OPEN, open);
    gpio_set(PIN_LED_LEAK, S.tripped && S.reason == MIRAGE_TRIP_LEAK);
    gpio_set(PIN_LED_GAS, fresh && S.gas_run > 0);
    gpio_set(PIN_LED_ALIVE, fresh && (S.n_verdicts / ALIVE_BLINK_VERDICTS) % 2 == 0);
    gpio_set(PIN_LED_READY, S.tripped && fresh && S.clear_run >= MIRAGE_RESET_CLEAR);
    gpio_set(PIN_LED_FAULT, S.tripped && S.reason != MIRAGE_TRIP_LEAK);
}

static void trip_locked(uint32_t reason, uint64_t t)
{
    if (S.tripped) {
        return;
    }
    S.tripped = 1;
    S.reason = reason;
    S.trip_ns = t;
    if (reason == MIRAGE_TRIP_LEAK && S.gas_start_frame_ns && t > S.gas_start_frame_ns) {
        S.detect_to_trip_ms = (float)(t - S.gas_start_frame_ns) / 1e6f;
    }
    apply_outputs_locked(t);
    fprintf(stderr, "[supervisor] TRIP (%s) at seq %u", reason_str(reason), S.last_seq);
    if (reason == MIRAGE_TRIP_LEAK) {
        fprintf(stderr, ", first gas frame to valve: %.1f ms", S.detect_to_trip_ms);
    }
    fprintf(stderr, "\n");
}

static void on_verdict(const mirage_verdict_t *v)
{
    const uint64_t t = now_ns();
    pthread_mutex_lock(&S.lock);
    S.last_verdict_ns = t;
    S.last_seq = v->seq;
    S.last_cls = v->cls;
    S.last_p_gas = v->p_gas;
    S.last_peak = v->peak_px;
    S.n_verdicts++;
    if (v->frame_ns && t > v->frame_ns) {
        S.lat_ms = (float)(t - v->frame_ns) / 1e6f;
        if (S.lat_ms > S.lat_max_ms) {
            S.lat_max_ms = S.lat_ms;
        }
    }
    const int gas = v->cls == MIRAGE_GAS || v->p_gas >= MIRAGE_P_GAS_TRIP;
    if (gas) {
        if (S.gas_run == 0) {
            S.gas_start_frame_ns = v->frame_ns;
        }
        S.gas_run++;
        S.clear_run = 0;
    } else {
        S.gas_run = 0;
        S.clear_run++;
    }
    if (S.gas_run >= MIRAGE_TRIP_RUN) {
        trip_locked(MIRAGE_TRIP_LEAK, t);
    }
    pthread_mutex_unlock(&S.lock);
}

static int try_reset_locked(uint64_t t)
{
    if (!S.tripped) {
        return 1;
    }
    const int fresh = S.last_verdict_ns &&
                      (t - S.last_verdict_ns) < (uint64_t)MIRAGE_DEADLINE_MS * 1000000ull;
    if (!fresh || S.clear_run < MIRAGE_RESET_CLEAR) {
        fprintf(stderr, "[supervisor] reset refused: %s\n",
                !fresh ? "vision is not sending verdicts" : "gas still in view");
        return 0;
    }
    S.tripped = 0;
    S.reason = MIRAGE_TRIP_NONE;
    S.gas_run = 0;
    apply_outputs_locked(t);
    fprintf(stderr, "[supervisor] reset accepted, valve open\n");
    return 1;
}

static void on_tick(void)
{
    const uint64_t t = now_ns();
    pthread_mutex_lock(&S.lock);
    if (!S.tripped && t - S.last_verdict_ns > (uint64_t)MIRAGE_DEADLINE_MS * 1000000ull) {
        trip_locked(MIRAGE_TRIP_DEADLINE, t);
    }
    if (S.reset_request) {
        S.reset_request = 0;
        try_reset_locked(t);
    }
    apply_outputs_locked(t);   /* cached, so only changed levels (blink, timeouts) are written */
    pthread_mutex_unlock(&S.lock);
}

static void fill_status_locked(mirage_status_t *st, int reset_ok)
{
    st->valve_open = !S.tripped;
    st->tripped = S.tripped;
    st->reason = S.reason;
    st->last_seq = S.last_seq;
    st->reset_ok = reset_ok;
}

/* ------------------------------------------------------------- telemetry */
/* Low priority on purpose: it can stall or die without touching the valve. */

static void *telemetry(void *arg)
{
    const char **cfg = (const char **)arg;     /* cfg[0] = ip or NULL, cfg[1] = port */
    int sock = -1;
    struct sockaddr_in dst;
    memset(&dst, 0, sizeof dst);
    if (cfg[0]) {
        sock = socket(AF_INET, SOCK_DGRAM, 0);
        dst.sin_family = AF_INET;
        dst.sin_port = htons((uint16_t)atoi(cfg[1]));
        if (sock == -1 || inet_pton(AF_INET, cfg[0], &dst.sin_addr) != 1) {
            fprintf(stderr, "[supervisor] telemetry disabled, bad address %s\n", cfg[0]);
            if (sock != -1) {
                close(sock);
            }
            sock = -1;
        }
    }
    int was_down = 1;                       /* a press only counts after a release has been seen */
    unsigned n = 0;
    for (;;) {
        usleep(50 * 1000);
        const int down = button_down();
        if (down && !was_down) {
            S.reset_request = 1;            /* handled by the supervisor thread on its next tick */
        }
        was_down = down;
        if (++n % 2) {
            continue;                       /* telemetry at 10 Hz */
        }
        char buf[512];
        pthread_mutex_lock(&S.lock);
        const uint64_t t = now_ns();
        const double since_ms = S.last_verdict_ns ? (double)(t - S.last_verdict_ns) / 1e6 : -1.0;
        const int len = snprintf(buf, sizeof buf,
            "{\"t_ms\":%llu,\"valve\":\"%s\",\"tripped\":%d,\"reason\":\"%s\",\"seq\":%u,"
            "\"cls\":%u,\"p_gas\":%.3f,\"peak_px\":%.3f,\"since_verdict_ms\":%.1f,"
            "\"lat_ms\":%.1f,\"lat_max_ms\":%.1f,\"detect_to_trip_ms\":%.1f,\"verdicts\":%u}",
            (unsigned long long)(t / 1000000ull), S.tripped ? "closed" : "open", S.tripped,
            reason_str(S.reason), S.last_seq, (unsigned)S.last_cls, S.last_p_gas, S.last_peak,
            since_ms, S.lat_ms, S.lat_max_ms, S.detect_to_trip_ms, S.n_verdicts);
        pthread_mutex_unlock(&S.lock);
        if (sock != -1 && len > 0) {
            sendto(sock, buf, (size_t)len, 0, (struct sockaddr *)&dst, sizeof dst);
        }
        if (n % 20 == 0) {
            printf("%s\n", buf);            /* once a second on the console */
            fflush(stdout);
        }
    }
    return NULL;
}

/* ------------------------------------------------------------------ main */

int main(int argc, char **argv)
{
    static const char *telem_cfg[2] = { NULL, "5005" };
    if (argc > 1) {
        telem_cfg[0] = argv[1];
    }
    if (argc > 2) {
        telem_cfg[1] = argv[2];
    }

    struct sched_param sp = { .sched_priority = SUP_PRIORITY };
    int rc = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
    if (rc != EOK) {
        fprintf(stderr, "[supervisor] warning: could not get SCHED_FIFO %d (%s), run as root\n",
                SUP_PRIORITY, strerror(rc));
    }

    gpio_setup();
    pthread_mutex_lock(&S.lock);
    apply_outputs_locked(now_ns());         /* boot state: valve closed */
    pthread_mutex_unlock(&S.lock);

    name_attach_t *att = name_attach(NULL, MIRAGE_SUP_NAME, 0);
    if (att == NULL) {
        perror("[supervisor] name_attach (is another supervisor already running?)");
        return EXIT_FAILURE;
    }

    /* A 10 ms timer delivers a pulse at our own priority: that is what enforces the deadline. */
    int self_coid = ConnectAttach(0, 0, att->chid, _NTO_SIDE_CHANNEL, 0);
    if (self_coid == -1) {
        perror("[supervisor] ConnectAttach");
        return EXIT_FAILURE;
    }
    struct sigevent ev;
    SIGEV_PULSE_INIT(&ev, self_coid, SUP_PRIORITY, PULSE_TICK, 0);
    timer_t tid;
    if (timer_create(CLOCK_MONOTONIC, &ev, &tid) == -1) {
        perror("[supervisor] timer_create");
        return EXIT_FAILURE;
    }
    struct itimerspec its;
    memset(&its, 0, sizeof its);
    its.it_value.tv_nsec = TICK_MS * 1000000L;
    its.it_interval.tv_nsec = TICK_MS * 1000000L;
    timer_settime(tid, 0, &its, NULL);

    pthread_attr_t ta;
    pthread_attr_init(&ta);
    pthread_attr_setinheritsched(&ta, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&ta, SCHED_RR);
    struct sched_param tsp = { .sched_priority = TELEM_PRIORITY };
    pthread_attr_setschedparam(&ta, &tsp);
    pthread_t th;
    if (pthread_create(&th, &ta, telemetry, (void *)telem_cfg) != EOK) {
        fprintf(stderr, "[supervisor] telemetry thread failed to start (the valve logic is unaffected)\n");
    }

    fprintf(stderr, "[supervisor] up as '%s' at priority %d. Valve CLOSED until reset.\n",
            MIRAGE_SUP_NAME, SUP_PRIORITY);

    union {
        struct _pulse    pulse;
        uint16_t         type;
        mirage_verdict_t verdict;
        mirage_cmd_t     cmd;
    } msg;

    for (;;) {
        const int rcvid = MsgReceive(att->chid, &msg, sizeof msg, NULL);
        if (rcvid == -1) {
            if (errno == EINTR) {
                continue;
            }
            perror("[supervisor] MsgReceive");
            break;
        }
        if (rcvid == 0) {                                   /* pulse */
            if (msg.pulse.code == PULSE_TICK) {
                on_tick();
            } else if (msg.pulse.code == _PULSE_CODE_DISCONNECT) {
                ConnectDetach(msg.pulse.scoid);
            }
            continue;
        }
        if (msg.type == _IO_CONNECT) {                      /* sent by name_open() */
            MsgReply(rcvid, EOK, NULL, 0);
            continue;
        }
        if (msg.type > _IO_BASE && msg.type <= _IO_MAX) {   /* other I/O messages */
            MsgError(rcvid, ENOSYS);
            continue;
        }
        int reset_ok = 0;
        if (msg.type == MIRAGE_MSG_VERDICT) {
            on_verdict(&msg.verdict);
        } else if (msg.type == MIRAGE_MSG_RESET) {
            pthread_mutex_lock(&S.lock);
            reset_ok = try_reset_locked(now_ns());
            pthread_mutex_unlock(&S.lock);
        } else if (msg.type != MIRAGE_MSG_STATUS) {
            MsgError(rcvid, ENOSYS);
            continue;
        }
        mirage_status_t st;
        pthread_mutex_lock(&S.lock);
        fill_status_locked(&st, reset_ok);
        pthread_mutex_unlock(&S.lock);
        MsgReply(rcvid, EOK, &st, sizeof st);
    }

    name_detach(att, 0);
    return EXIT_FAILURE;
}
