#include "lighting.h"
#include "board.h"
void Lighting_Apply(uint8_t percent)
{
    Board_SetPwm(percent <= 100U ? percent : 0U);
}
