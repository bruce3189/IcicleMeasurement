#include "app_control.h"
#include <stddef.h>

static void ClearResult(AppControl *app)
{
    app->has_result = false;
    app->distance_tenth_mm = 0U;
}

static void StopLocally(AppControl *app)
{
    app->light_percent = 0U;
    app->synchronized = false;
    app->start_pending = false;
    if (app->state != APP_FAULT) app->state = APP_STOPPED;
    ClearResult(app);
}

static bool Send(AppControl *app, AppRequest request)
{
    return app->send != NULL && app->send(request, app->send_context);
}

void App_Init(AppControl *app, AppSendRequest send, void *context)
{
    *app = (AppControl){0};
    app->state = APP_STOPPED;
    app->send = send;
    app->send_context = context;
}

void App_TransportLost(AppControl *app)
{
    app->connected = false;
    StopLocally(app);
}

void App_Tick(AppControl *app, uint32_t now_ms)
{
    /* Unsigned subtraction also handles the HAL millisecond counter wrapping. */
    if (app->connected &&
        (uint32_t)(now_ms - app->last_heartbeat_ms) >= APP_HEARTBEAT_TIMEOUT_MS) {
        App_TransportLost(app);
    }
}

void App_SetStopHeld(AppControl *app, bool held)
{
    app->stop_held = held;
    if (held) StopLocally(app);
}

bool App_RequestStart(AppControl *app)
{
    if (!app->connected || !app->synchronized || app->stop_held ||
        app->state != APP_READY || app->start_pending) return false;
    if (!Send(app, APP_REQUEST_START)) return false;
    app->start_pending = true;
    ClearResult(app);
    /* Host STATE(MEASURING), not the button, confirms the transition. */
    return true;
}

void App_RequestStop(AppControl *app)
{
    StopLocally(app);
    (void)Send(app, APP_REQUEST_STOP);
}

bool App_HandleHostEvent(AppControl *app, const AppEvent *event, uint32_t now_ms)
{
    App_Tick(app, now_ms);
    if (event->kind == APP_HOST_HEARTBEAT) {
        app->connected = true;
        app->last_heartbeat_ms = now_ms;
        (void)Send(app, APP_REQUEST_HEARTBEAT_ACK);
        return true; /* A heartbeat alone never synchronizes state. */
    }
    if (!app->connected) return false;
    if (event->kind == APP_HOST_FAULT) {
        StopLocally(app);
        app->state = APP_FAULT;
        return true;
    }
    if (event->kind == APP_HOST_SYNC) {
        if (app->stop_held || (unsigned)event->state > (unsigned)APP_FAULT)
            return false;
        /* A fresh handshake must establish this event, never an old queued frame. */
        app->state = event->state;
        app->synchronized = event->state != APP_FAULT;
        app->start_pending = false;
        app->light_percent = 0U;
        ClearResult(app);
        return true;
    }
    if (!app->synchronized || app->stop_held || app->state == APP_FAULT) return false;
    switch (event->kind) {
    case APP_HOST_STATE:
        if ((unsigned)event->state > (unsigned)APP_FAULT) return false;
        if (event->state == APP_FAULT || event->state == APP_STOPPED) {
            StopLocally(app);
        } else if (event->state == APP_MEASURING && app->state != APP_MEASURING) {
            ClearResult(app);
        }
        app->state = event->state;
        app->start_pending = false;
        return true;
    case APP_HOST_RESULT:
        /* The adapter must correlate the result with the current task. */
        if (app->state != APP_MEASURING) return false;
        app->distance_tenth_mm = event->value;
        app->has_result = true;
        return true;
    case APP_HOST_REJECT:
        ClearResult(app);
        app->start_pending = false;
        app->state = APP_READY;
        return true;
    case APP_HOST_CANCEL:
        StopLocally(app);
        return true;
    case APP_HOST_LIGHT_SET:
        if (event->value > 100U || app->state == APP_STOPPED) return false;
        app->light_percent = (uint8_t)event->value;
        return true;
    default:
        return false;
    }
}
