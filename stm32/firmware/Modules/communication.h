#ifndef COMMUNICATION_H
#define COMMUNICATION_H
#include "app_control.h"
typedef struct { uint32_t rx_overflows, uart_errors, restarts_failed; } CommunicationStats;
extern volatile CommunicationStats communication_stats;
bool Communication_Init(void);
void Communication_Poll(AppControl *app, uint32_t now_ms);
void Communication_ResetRx(void);
/* Main-loop only. Copies <=64 bytes; false means busy/invalid/not sent. */
bool Communication_WriteFrame(const uint8_t *data, uint16_t size);
#endif
