#include "board.h"
#include "app_control.h"
#include "buttons.h"
#include "communication.h"
#include "display.h"
#include "lighting.h"
#include "protocol.h"
#include "status_led.h"

AppControl controller;
static Buttons buttons;
static bool SendRequest(AppRequest request, void *context)
{
    (void)context;
    return Protocol_Send(request);
}
static void ApplyOutputs(void)
{
    Lighting_Apply(controller.light_percent);
    StatusLed_Update(controller.connected, controller.state == APP_FAULT);
}

int main(void)
{
    HAL_Init();
    if (!Board_Init()) Board_Panic();
    App_Init(&controller, SendRequest, NULL);
    Buttons_Init(&buttons);
    Protocol_Init();
    if (!Communication_Init()) Board_Panic();
    Display_Init();
    for (;;) {
        uint32_t now = HAL_GetTick();
        bool was_connected = controller.connected;
        bool stop = Board_StopPressed();
        App_Tick(&controller, now);
        if (was_connected && !controller.connected) Communication_ResetRx();
        App_SetStopHeld(&controller, stop);
        ApplyOutputs(); /* Local light-off happens before any transport/display. */
        uint8_t events = Buttons_Update(&buttons, Board_StartPressed(), stop, now);
        if (events & BUTTON_STOP) {
            Communication_ResetRx(); /* Discard queued bytes/partial old command. */
            App_RequestStop(&controller);
        } else if (!stop && (events & BUTTON_START)) {
            (void)App_RequestStart(&controller);
        }
        Communication_Poll(&controller, now);
        App_Tick(&controller, HAL_GetTick());
        ApplyOutputs();
        Display_Update(&controller, HAL_GetTick());
        HAL_Delay(1U);
    }
}
