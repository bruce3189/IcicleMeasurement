#include "buttons.h"
void Buttons_Init(Buttons *buttons) { *buttons = (Buttons){0}; }
static bool Pressed(ButtonFilter *key, bool raw, uint32_t now_ms)
{
    if (raw != key->candidate) { key->candidate = raw; key->changed_ms = now_ms; }
    if (key->stable != key->candidate &&
        (uint32_t)(now_ms - key->changed_ms) >= BUTTON_DEBOUNCE_MS) {
        key->stable = key->candidate;
        return key->stable;
    }
    return false;
}
uint8_t Buttons_Update(Buttons *buttons, bool start, bool stop, uint32_t now_ms)
{
    uint8_t events = 0U;
    if (Pressed(&buttons->start, start, now_ms)) events |= BUTTON_START;
    if (Pressed(&buttons->stop, stop, now_ms)) events |= BUTTON_STOP;
    return events;
}
