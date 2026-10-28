#include "protocol.h"

ProtocolStats protocol_stats;
void Protocol_Init(void) { protocol_stats = (ProtocolStats){0}; }
void Protocol_Reset(void) { /* Clear the future decoder's partial frame/session. */ }
bool Protocol_IsReady(void) { return false; }

void Protocol_Feed(uint8_t byte, AppControl *app, uint32_t now_ms)
{
    /* TODO(protocol): implement agreed framing, checksum and task/session
     * correlation; only then call App_HandleHostEvent. Never execute raw bytes.
     * Until implemented, neither real nor simulated link-up is reported. */
    (void)byte; (void)app; (void)now_ms;
    ++protocol_stats.discarded_bytes;
}

bool Protocol_Send(AppRequest request)
{
    /* TODO(protocol): encode START/STOP/ACK then Communication_WriteFrame.
     * This stub explicitly reports failure; it never pretends to send. */
    (void)request;
    ++protocol_stats.unencoded_requests;
    return false;
}
