#ifndef PROTOCOL_H
#define PROTOCOL_H
#include "app_control.h"
typedef struct { uint32_t discarded_bytes, unencoded_requests; } ProtocolStats;
extern ProtocolStats protocol_stats;
void Protocol_Init(void);
void Protocol_Reset(void);
void Protocol_Feed(uint8_t byte, AppControl *app, uint32_t now_ms);
bool Protocol_Send(AppRequest request);
bool Protocol_IsReady(void);
#endif
