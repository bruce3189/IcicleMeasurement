#ifndef APP_CONTROL_H
#define APP_CONTROL_H

#include <stdbool.h>
#include <stdint.h>

#define APP_HEARTBEAT_TIMEOUT_MS 3000U

typedef enum { APP_READY, APP_MEASURING, APP_STOPPED, APP_FAULT } AppState;
typedef enum { APP_REQUEST_START, APP_REQUEST_STOP, APP_REQUEST_HEARTBEAT_ACK } AppRequest;
typedef bool (*AppSendRequest)(AppRequest request, void *context);

/* Only the protocol adapter may emit these AFTER complete-frame/checksum,
 * session and current-task validation. These are C events, NOT wire opcodes. */
typedef enum {
    APP_HOST_HEARTBEAT, APP_HOST_SYNC, APP_HOST_STATE, APP_HOST_RESULT,
    APP_HOST_REJECT, APP_HOST_CANCEL, APP_HOST_LIGHT_SET, APP_HOST_FAULT
} AppEventKind;

typedef struct {
    AppEventKind kind;
    AppState state;             /* SYNC / STATE */
    uint32_t value;             /* RESULT: 0.1 mm units; LIGHT_SET: 0..100 */
} AppEvent;

typedef struct {
    AppState state;
    bool connected;
    bool synchronized;
    bool start_pending;
    bool stop_held;
    bool has_result;
    uint8_t light_percent;
    uint32_t distance_tenth_mm;
    uint32_t last_heartbeat_ms;
    AppSendRequest send;
    void *send_context;
} AppControl;

void App_Init(AppControl *app, AppSendRequest send, void *context);
void App_Tick(AppControl *app, uint32_t now_ms);
bool App_HandleHostEvent(AppControl *app, const AppEvent *event, uint32_t now_ms);
bool App_RequestStart(AppControl *app);
void App_RequestStop(AppControl *app);
/* Raw STOP gates light immediately; debounced press sends just one request. */
void App_SetStopHeld(AppControl *app, bool held);
void App_TransportLost(AppControl *app);

#endif
