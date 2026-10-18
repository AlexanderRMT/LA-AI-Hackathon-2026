// verdict_client.hpp: sends verdicts to the supervisor over QNX message passing.   owner: Jaime
#pragma once

#include <cstdint>
#include <sys/dispatch.h>
#include <sys/neutrino.h>

extern "C" {
#include "../common/mirage_msgs.h"
}

class VerdictClient {
public:
    ~VerdictClient()
    {
        if (coid_ != -1) {
            name_close(coid_);
        }
    }

    // Returns false if the supervisor can't be reached. That's safe: it fails closed on its own.
    bool send(mirage_verdict_t& v, mirage_status_t* status = nullptr)
    {
        if (coid_ == -1) {
            coid_ = name_open(MIRAGE_SUP_NAME, 0);
            if (coid_ == -1) {
                return false;
            }
        }
        mirage_status_t tmp;
        v.type = MIRAGE_MSG_VERDICT;
        v.subtype = 0;
        v.seq = ++seq_;
        if (MsgSend(coid_, &v, sizeof v, status ? status : &tmp, sizeof tmp) == -1) {
            name_close(coid_);
            coid_ = -1;
            return false;
        }
        return true;
    }

private:
    int coid_ = -1;
    uint32_t seq_ = 0;
};
