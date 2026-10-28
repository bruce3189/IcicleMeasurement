#ifndef OLED_PORT_H
#define OLED_PORT_H
#include "display.h"
bool OledPort_Init(void);
/* Return promptly; a future driver must send bounded chunks, not block a frame. */
bool OledPort_Draw(const DisplayFrame *frame);
#endif
