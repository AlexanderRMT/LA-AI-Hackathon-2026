/*
 * mirage_ctl.c: ask the supervisor for its status, or request a reset.
 * usage: mirage_ctl status | reset
 * Build: qcc -Vgcc_ntoaarch64le -O2 -o mirage_ctl mirage_ctl.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/dispatch.h>
#include <sys/neutrino.h>
#include "../common/mirage_msgs.h"

int main(int argc, char **argv)
{
    if (argc < 2 || (strcmp(argv[1], "status") && strcmp(argv[1], "reset"))) {
        fprintf(stderr, "usage: %s status | reset\n", argv[0]);
        return EXIT_FAILURE;
    }
    int coid = name_open(MIRAGE_SUP_NAME, 0);
    if (coid == -1) {
        perror("name_open (is the supervisor running?)");
        return EXIT_FAILURE;
    }
    const int reset = strcmp(argv[1], "reset") == 0;
    mirage_cmd_t cmd = { (uint16_t)(reset ? MIRAGE_MSG_RESET : MIRAGE_MSG_STATUS), 0 };
    mirage_status_t st;
    if (MsgSend(coid, &cmd, sizeof cmd, &st, sizeof st) == -1) {
        perror("MsgSend");
        return EXIT_FAILURE;
    }
    printf("valve %s, tripped %d, reason %u, last seq %u%s\n",
           st.valve_open ? "OPEN" : "CLOSED", st.tripped, st.reason, st.last_seq,
           reset ? (st.reset_ok ? ", reset accepted" : ", reset refused") : "");
    name_close(coid);
    return st.valve_open ? EXIT_SUCCESS : 2;
}
