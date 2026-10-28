#ifndef DISPLAY_H
#define DISPLAY_H
#include "app_control.h"
typedef struct { char rows[3][32]; } DisplayFrame;
/* Exposed for debugger inspection even when the OLED port is unimplemented. */
extern DisplayFrame display_frame;
void Display_Init(void);
void Display_Update(const AppControl *app, uint32_t now_ms);
void Display_Format(const AppControl *app, DisplayFrame *frame);
#endif
