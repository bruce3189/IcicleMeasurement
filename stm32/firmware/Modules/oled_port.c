#include "oled_port.h"

/* Integration point: confirm controller, I2C address, supply and pull-ups first.
 * No SSD1306/SH1106 commands are sent by this scaffold. False means NOT READY. */
bool OledPort_Init(void) { return false; }
bool OledPort_Draw(const DisplayFrame *frame)
{
    (void)frame;
    return false;
}
