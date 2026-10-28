/* Portable host tests. Requires a native C compiler, not arm-none-eabi-gcc.
 * The target build can syntax-check these, but that is not test execution. */
#include "app_control.h"
#include "buttons.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

static bool tx_ok;
static unsigned sent[3];
static bool Capture(AppRequest request, void *context)
{
    (void)context;
    ++sent[request];
    return tx_ok;
}
static bool Event(AppControl *app, AppEventKind kind, AppState state, uint32_t value, uint32_t now)
{
    AppEvent event = {kind, state, value};
    return App_HandleHostEvent(app, &event, now);
}
static void Ready(AppControl *app, uint32_t now)
{
    assert(Event(app, APP_HOST_HEARTBEAT, APP_READY, 0U, now));
    assert(Event(app, APP_HOST_SYNC, APP_READY, 0U, now));
}

int main(void)
{
    AppControl app;
    Buttons buttons;
    tx_ok = true;
    App_Init(&app, Capture, NULL);
    assert(!app.connected && !app.has_result && app.light_percent == 0U);
    assert(!App_RequestStart(&app));
    assert(!Event(&app, APP_HOST_LIGHT_SET, APP_READY, 50U, 0U));

    assert(Event(&app, APP_HOST_HEARTBEAT, APP_READY, 0U, 10U));
    assert(sent[APP_REQUEST_HEARTBEAT_ACK] == 1U);
    assert(!app.synchronized && !App_RequestStart(&app));
    assert(Event(&app, APP_HOST_SYNC, APP_READY, 0U, 10U));
    tx_ok = false;
    assert(!App_RequestStart(&app) && !app.start_pending);
    tx_ok = true;
    assert(App_RequestStart(&app));
    assert(app.state == APP_READY && app.start_pending);
    assert(!App_RequestStart(&app));
    assert(Event(&app, APP_HOST_STATE, APP_MEASURING, 0U, 11U));
    assert(Event(&app, APP_HOST_RESULT, APP_READY, 1234U, 12U));
    assert(app.has_result && app.distance_tenth_mm == 1234U);
    assert(Event(&app, APP_HOST_LIGHT_SET, APP_READY, 50U, 12U));
    assert(!Event(&app, APP_HOST_LIGHT_SET, APP_READY, 101U, 12U));
    assert(app.light_percent == 50U);

    tx_ok = false; /* STOP still turns off when transport cannot send. */
    App_RequestStop(&app);
    assert(app.light_percent == 0U && !app.has_result && !app.synchronized);
    assert(!Event(&app, APP_HOST_LIGHT_SET, APP_READY, 100U, 13U));
    tx_ok = true;
    App_SetStopHeld(&app, true);
    assert(!Event(&app, APP_HOST_SYNC, APP_READY, 0U, 13U));
    App_SetStopHeld(&app, false);
    Ready(&app, 20U);
    assert(app.light_percent == 0U && !app.has_result);
    assert(Event(&app, APP_HOST_LIGHT_SET, APP_READY, 100U, 20U));
    App_Tick(&app, 3019U);
    assert(app.connected && app.light_percent == 100U);
    App_Tick(&app, 3020U);
    assert(!app.connected && app.light_percent == 0U && !app.has_result);
    assert(Event(&app, APP_HOST_HEARTBEAT, APP_READY, 0U, 3021U));
    assert(!app.synchronized && !Event(&app, APP_HOST_LIGHT_SET, APP_READY, 50U, 3021U));

    Ready(&app, 4000U);
    assert(Event(&app, APP_HOST_STATE, APP_MEASURING, 0U, 4000U));
    assert(Event(&app, APP_HOST_RESULT, APP_READY, 1234U, 4000U));
    assert(Event(&app, APP_HOST_REJECT, APP_READY, 0U, 4000U));
    assert(!app.has_result && app.state == APP_READY);
    assert(Event(&app, APP_HOST_LIGHT_SET, APP_READY, 50U, 4000U));
    assert(Event(&app, APP_HOST_FAULT, APP_READY, 0U, 4000U));
    assert(app.state == APP_FAULT && app.light_percent == 0U);
    assert(Event(&app, APP_HOST_HEARTBEAT, APP_READY, 0U, 4001U));
    assert(app.state == APP_FAULT && !app.synchronized);
    assert(!Event(&app, APP_HOST_STATE, APP_READY, 0U, 4001U));
    Ready(&app, 4002U);
    assert(Event(&app, APP_HOST_CANCEL, APP_READY, 0U, 4002U));
    assert(!app.has_result && app.state == APP_STOPPED && app.light_percent == 0U);

    /* Tick wrap: elapsed is 2999, then exactly 3000, even across UINT32_MAX. */
    App_Init(&app, Capture, NULL);
    Ready(&app, UINT32_MAX - 1000U);
    App_Tick(&app, 1998U);
    assert(app.connected);
    App_Tick(&app, 1999U);
    assert(!app.connected);

    Buttons_Init(&buttons);
    assert(Buttons_Update(&buttons, true, false, 10U) == 0U);
    assert(Buttons_Update(&buttons, false, false, 15U) == 0U);
    assert(Buttons_Update(&buttons, true, false, 20U) == 0U);
    assert(Buttons_Update(&buttons, true, false, 49U) == 0U);
    assert(Buttons_Update(&buttons, true, false, 50U) == BUTTON_START);
    assert(Buttons_Update(&buttons, true, false, 500U) == 0U);
    assert(Buttons_Update(&buttons, false, false, 501U) == 0U);
    assert(Buttons_Update(&buttons, false, false, 531U) == 0U);
    assert(Buttons_Update(&buttons, true, true, 532U) == 0U);
    assert(Buttons_Update(&buttons, true, true, 562U) == (BUTTON_START | BUTTON_STOP));
    puts("PASS: state transitions, stop, timeout, reconnect, faults, wrap, debounce");
    return 0;
}
