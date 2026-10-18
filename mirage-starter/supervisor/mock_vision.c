/*
 * mock_vision.c: stands in for the vision process so the supervisor can be built
 * and tested without a camera. Sends a verdict every 100 ms.      owner: Alex
 *
 * Keys: c clear   g gas   h heat   s stall/resume   x ask for reset   q quit
 * Build: qcc -Vgcc_ntoaarch64le -O2 -o mock_vision mock_vision.c    (or -Vgcc_ntox86_64 for the VM)
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#include <sys/select.h>
#include <sys/dispatch.h>
#include <sys/neutrino.h>
#include "../common/mirage_msgs.h"

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int main(void)
{
    int coid = name_open(MIRAGE_SUP_NAME, 0);
    if (coid == -1) {
        perror("name_open (is the supervisor running?)");
        return EXIT_FAILURE;
    }
    struct termios old, raw;
    const int tty = isatty(STDIN_FILENO);
    if (tty) {
        tcgetattr(STDIN_FILENO, &old);
        raw = old;
        raw.c_lflag &= ~(tcflag_t)(ICANON | ECHO);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    }
    printf("mock_vision: c clear, g gas, h heat, s stall/resume, x reset, q quit\n");
    int mode = MIRAGE_CLEAR, stalled = 0, running = 1, last_open = -1;
    uint32_t seq = 0;
    while (running) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(STDIN_FILENO, &fds);
        struct timeval tv = { 0, 100000 };
        if (select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0) {
            char ch;
            if (read(STDIN_FILENO, &ch, 1) == 1) {
                switch (ch) {
                case 'c': mode = MIRAGE_CLEAR; printf("sending clear\n"); break;
                case 'g': mode = MIRAGE_GAS;   printf("sending gas\n");   break;
                case 'h': mode = MIRAGE_HEAT;  printf("sending heat\n");  break;
                case 's': stalled = !stalled;  printf(stalled ? "stalled, nothing sent\n" : "resumed\n"); break;
                case 'x': {
                    mirage_cmd_t cmd = { MIRAGE_MSG_RESET, 0 };
                    mirage_status_t st;
                    if (MsgSend(coid, &cmd, sizeof cmd, &st, sizeof st) != -1) {
                        printf("reset %s\n", st.reset_ok ? "accepted" : "refused");
                    }
                    break;
                }
                case 'q': running = 0; break;
                default: break;
                }
                fflush(stdout);
            }
        }
        if (stalled || !running) {
            continue;
        }
        mirage_verdict_t v;
        memset(&v, 0, sizeof v);
        v.type = MIRAGE_MSG_VERDICT;
        v.seq = ++seq;
        v.frame_ns = now_ns() - 30000000ull;        /* pretend the frame is 30 ms old */
        v.sent_ns = now_ns();
        v.cls = (uint8_t)mode;
        v.p_gas = mode == MIRAGE_GAS ? 0.95f : 0.02f;
        v.conf = 0.95f;
        v.peak_px = mode == MIRAGE_CLEAR ? 0.01f : 0.6f;
        mirage_status_t st;
        if (MsgSend(coid, &v, sizeof v, &st, sizeof st) == -1) {
            perror("MsgSend");
            break;
        }
        if (st.valve_open != last_open) {
            printf("valve %s (reason %u)\n", st.valve_open ? "OPEN" : "CLOSED", st.reason);
            fflush(stdout);
            last_open = st.valve_open;
        }
    }
    if (tty) {
        tcsetattr(STDIN_FILENO, TCSANOW, &old);
    }
    name_close(coid);
    return EXIT_SUCCESS;
}
