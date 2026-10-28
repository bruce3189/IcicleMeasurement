#include "display.h"
#include "oled_port.h"
#include <string.h>

DisplayFrame display_frame;
static uint32_t last_refresh;

static char *Decimal(char *out, uint32_t value)
{
    char reverse[10];
    unsigned count = 0U;
    do { reverse[count++] = (char)('0' + value % 10U); value /= 10U; } while (value);
    while (count) *out++ = reverse[--count];
    *out = '\0';
    return out;
}

void Display_Format(const AppControl *app, DisplayFrame *frame)
{
    const char *states[] = {"STATE READY", "STATE MEASURING", "STATE STOPPED", "STATE FAULT"};
    memset(frame, 0, sizeof(*frame));
    strcpy(frame->rows[0], app->connected ? "LINK OK" : "LINK LOST");
    strcpy(frame->rows[1], states[app->state]);
    if (!app->has_result) { strcpy(frame->rows[2], "NO RESULT"); return; }
    char *next = frame->rows[2];
    *next++ = 'L'; *next++ = '=';
    next = Decimal(next, app->distance_tenth_mm / 10U);
    *next++ = '.';
    *next++ = (char)('0' + app->distance_tenth_mm % 10U);
    strcpy(next, " mm");
}

void Display_Init(void)
{
    last_refresh = (uint32_t)(0U - 100U);
    (void)OledPort_Init();
}

void Display_Update(const AppControl *app, uint32_t now_ms)
{
    if ((uint32_t)(now_ms - last_refresh) < 100U) return;
    last_refresh = now_ms;
    Display_Format(app, &display_frame);
    (void)OledPort_Draw(&display_frame);
}
