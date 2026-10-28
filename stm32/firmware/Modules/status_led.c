#include "status_led.h"
#include "board.h"
void StatusLed_Update(bool connected, bool fault)
{
    bool normal = connected && !fault;
    Board_SetLeds(normal, !normal);
}
