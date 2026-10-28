#include "board.h"
void SysTick_Handler(void) { HAL_IncTick(); }
void USART1_IRQHandler(void) { HAL_UART_IRQHandler(&huart1); }
void HardFault_Handler(void) { Board_Panic(); }
void MemManage_Handler(void) { Board_Panic(); }
void BusFault_Handler(void) { Board_Panic(); }
void UsageFault_Handler(void) { Board_Panic(); }
void NMI_Handler(void) { Board_Panic(); }
