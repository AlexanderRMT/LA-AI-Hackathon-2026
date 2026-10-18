/*
 * mirage_msgs.h: the contract between the vision process and the supervisor.
 * Change it only with all three of you in the loop; both sides must rebuild.
 */
#ifndef MIRAGE_MSGS_H
#define MIRAGE_MSGS_H

#include <stdint.h>
#include <sys/iomsg.h>

#define MIRAGE_SUP_NAME      "mirage_supervisor"   /* name_attach() / name_open() name */

/* Custom message types sit above the QNX I/O message range. */
#define MIRAGE_MSG_VERDICT   (_IO_MAX + 0x10)
#define MIRAGE_MSG_RESET     (_IO_MAX + 0x11)
#define MIRAGE_MSG_STATUS    (_IO_MAX + 0x12)

/* Model classes, in model output order. Keep in sync with mac/train.py LABELS. */
enum mirage_class { MIRAGE_CLEAR = 0, MIRAGE_GAS = 1, MIRAGE_HEAT = 2, MIRAGE_MOTION = 3 };

enum mirage_trip { MIRAGE_TRIP_NONE = 0, MIRAGE_TRIP_BOOT = 1, MIRAGE_TRIP_LEAK = 2, MIRAGE_TRIP_DEADLINE = 3 };

#define MIRAGE_DEADLINE_MS   250    /* no verdict for this long: fail closed */
#define MIRAGE_TRIP_RUN      3      /* consecutive gas verdicts needed to trip */
#define MIRAGE_RESET_CLEAR   5      /* consecutive clear verdicts needed before a reset is accepted */
#define MIRAGE_P_GAS_TRIP    0.60f  /* a verdict counts as gas at or above this probability */

typedef struct {
    uint16_t type;        /* MIRAGE_MSG_VERDICT */
    uint16_t subtype;     /* 0 */
    uint32_t seq;         /* increments with every verdict */
    uint64_t frame_ns;    /* CLOCK_MONOTONIC time of the newest frame behind this verdict */
    uint64_t sent_ns;     /* CLOCK_MONOTONIC time the verdict was sent */
    uint8_t  cls;         /* enum mirage_class with the highest probability */
    uint8_t  reserved[3];
    float    p_gas;       /* model probability of MIRAGE_GAS */
    float    conf;        /* probability of cls */
    float    peak_px;     /* peak dot shift after bump removal, in pixels */
} mirage_verdict_t;

typedef struct {
    uint16_t type;        /* MIRAGE_MSG_RESET or MIRAGE_MSG_STATUS */
    uint16_t subtype;     /* 0 */
} mirage_cmd_t;

typedef struct {
    int32_t  valve_open;  /* 1 while the relay is energized and gas can flow */
    int32_t  tripped;     /* 1 while latched closed */
    uint32_t reason;      /* enum mirage_trip */
    uint32_t last_seq;    /* seq of the last verdict received */
    int32_t  reset_ok;    /* in a reply to MIRAGE_MSG_RESET: 1 accepted, 0 refused */
} mirage_status_t;

#endif /* MIRAGE_MSGS_H */
