#ifndef BUTTONS_H
#define BUTTONS_H
#include <stdbool.h>
#include <stdint.h>
#define BUTTON_START 1U
#define BUTTON_STOP  2U
#define BUTTON_DEBOUNCE_MS 30U
typedef struct { bool candidate, stable; uint32_t changed_ms; } ButtonFilter;
typedef struct { ButtonFilter start, stop; } Buttons;
void Buttons_Init(Buttons *buttons);
uint8_t Buttons_Update(Buttons *buttons, bool start_pressed, bool stop_pressed, uint32_t now_ms);
#endif
