#ifndef BOARD_H
#define BOARD_H
#include <stdbool.h>
#include "stm32f1xx_hal.h"

#define BOARD_UART_TX_PIN GPIO_PIN_9
#define BOARD_UART_RX_PIN GPIO_PIN_10
#define BOARD_OLED_SCL_PIN GPIO_PIN_6
#define BOARD_OLED_SDA_PIN GPIO_PIN_7
#define BOARD_GREEN_PIN GPIO_PIN_8
#define BOARD_RED_PIN GPIO_PIN_9
#define BOARD_START_PIN GPIO_PIN_12
#define BOARD_STOP_PIN GPIO_PIN_13
#define BOARD_PWM_PIN GPIO_PIN_6
#define BOARD_PWM_PERIOD 1000U

extern UART_HandleTypeDef huart1;
bool Board_Init(void);
bool Board_StartPressed(void);
bool Board_StopPressed(void);
void Board_SetLeds(bool green, bool red);
void Board_SetPwm(uint8_t percent);
/* Future OLED driver: 7-bit address; at most 32 bytes / 5 ms per call. */
bool Board_I2cWrite(uint8_t address7, const uint8_t *data, uint16_t size);
_Noreturn void Board_Panic(void);
#endif
