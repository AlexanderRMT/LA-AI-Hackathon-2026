/*
 * cpu_hog.c: burns CPU at a chosen priority for the chaos demo.     owner: Alex
 * usage: cpu_hog <priority> <threads> <seconds>
 *   cpu_hog 15 4 10   below vision (20): Mirage keeps up, nothing trips
 *   cpu_hog 30 4 10   above vision, below the supervisor (50): vision starves and
 *                     the supervisor fails closed within the 250 ms deadline
 * It quits on its own, because a priority-30 hog on every core also freezes your
 * shell and SSH session until it ends. Run as root for priorities above your limit.
 * Build: qcc -Vgcc_ntoaarch64le -O2 -o cpu_hog cpu_hog.c
 */
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double end_s;

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void *spin(void *arg)
{
    (void)arg;
    volatile unsigned long x = 0;
    while (now_s() < end_s) {
        x++;
    }
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <priority> <threads> <seconds>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int prio = atoi(argv[1]);
    int n = atoi(argv[2]);
    const double secs = atof(argv[3]);
    if (n < 1 || n > 64) {
        n = 4;
    }
    end_s = now_s() + secs;
    pthread_attr_t a;
    pthread_attr_init(&a);
    pthread_attr_setinheritsched(&a, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&a, SCHED_FIFO);
    struct sched_param sp = { .sched_priority = prio };
    pthread_attr_setschedparam(&a, &sp);
    printf("hogging %d threads at priority %d for %.0f s\n", n, prio, secs);
    fflush(stdout);
    pthread_t t[64];
    for (int i = 0; i < n; i++) {
        const int rc = pthread_create(&t[i], &a, spin, NULL);
        if (rc != 0) {
            fprintf(stderr, "pthread_create: %s (run as root for this priority)\n", strerror(rc));
            return EXIT_FAILURE;
        }
    }
    for (int i = 0; i < n; i++) {
        pthread_join(t[i], NULL);
    }
    return EXIT_SUCCESS;
}
