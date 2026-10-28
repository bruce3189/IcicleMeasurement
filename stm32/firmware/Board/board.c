#include "board.h"

UART_HandleTypeDef huart1;
static I2C_HandleTypeDef hi2c1;
static TIM_HandleTypeDef htim3;

static void SafeOutputs(void)
{
    GPIO_InitTypeDef pins = {0};
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    HAL_GPIO_WritePin(GPIOA, BOARD_PWM_PIN, GPIO_PIN_RESET);
    pins.Pin = BOARD_PWM_PIN;
    pins.Mode = GPIO_MODE_OUTPUT_PP;
    pins.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOA, &pins);
    HAL_GPIO_WritePin(GPIOB, BOARD_GREEN_PIN, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(GPIOB, BOARD_RED_PIN, GPIO_PIN_SET);
    pins.Pin = BOARD_GREEN_PIN | BOARD_RED_PIN;
    HAL_GPIO_Init(GPIOB, &pins);
}

static bool Clock_Init(void)
{
    RCC_OscInitTypeDef oscillator = {0};
    RCC_ClkInitTypeDef clocks = {0};
    oscillator.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    oscillator.HSIState = RCC_HSI_ON;
    oscillator.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    oscillator.PLL.PLLState = RCC_PLL_OFF;
    if (HAL_RCC_OscConfig(&oscillator) != HAL_OK) return false;
    clocks.ClockType = RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK |
                       RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    clocks.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
    clocks.AHBCLKDivider = RCC_SYSCLK_DIV1;
    clocks.APB1CLKDivider = RCC_HCLK_DIV1;
    clocks.APB2CLKDivider = RCC_HCLK_DIV1;
    return HAL_RCC_ClockConfig(&clocks, FLASH_LATENCY_0) == HAL_OK;
}

static bool Peripherals_Init(void)
{
    GPIO_InitTypeDef pins = {0};
    TIM_OC_InitTypeDef channel = {0};
    __HAL_RCC_AFIO_CLK_ENABLE();
    __HAL_RCC_USART1_CLK_ENABLE();
    __HAL_RCC_I2C1_CLK_ENABLE();
    __HAL_RCC_TIM3_CLK_ENABLE();
    /* Default mappings; PA13/PA14 remain available for SWD. */
    pins.Pin = BOARD_START_PIN | BOARD_STOP_PIN;
    pins.Mode = GPIO_MODE_INPUT;
    pins.Pull = GPIO_PULLUP;
    HAL_GPIO_Init(GPIOB, &pins);
    pins.Pin = BOARD_UART_TX_PIN;
    pins.Mode = GPIO_MODE_AF_PP;
    pins.Pull = GPIO_NOPULL;
    pins.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOA, &pins);
    pins.Pin = BOARD_UART_RX_PIN;
    pins.Mode = GPIO_MODE_INPUT;
    pins.Pull = GPIO_PULLUP;
    HAL_GPIO_Init(GPIOA, &pins);
    huart1.Instance = USART1;
    huart1.Init.BaudRate = 115200U;
    huart1.Init.WordLength = UART_WORDLENGTH_8B;
    huart1.Init.StopBits = UART_STOPBITS_1;
    huart1.Init.Parity = UART_PARITY_NONE;
    huart1.Init.Mode = UART_MODE_TX_RX;
    huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart1.Init.OverSampling = UART_OVERSAMPLING_16;
    if (HAL_UART_Init(&huart1) != HAL_OK) return false;
    HAL_NVIC_SetPriority(USART1_IRQn, 1U, 0U);
    HAL_NVIC_EnableIRQ(USART1_IRQn);

    pins.Pin = BOARD_OLED_SCL_PIN | BOARD_OLED_SDA_PIN;
    pins.Mode = GPIO_MODE_AF_OD;
    pins.Pull = GPIO_NOPULL; /* Actual module must provide 3.3 V pull-ups. */
    HAL_GPIO_Init(GPIOB, &pins);
    hi2c1.Instance = I2C1;
    hi2c1.Init.ClockSpeed = 100000U;
    hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
    hi2c1.Init.OwnAddress1 = 0U;
    hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
    hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
    if (HAL_I2C_Init(&hi2c1) != HAL_OK) return false;

    htim3.Instance = TIM3;
    /* HSI 8 MHz / (7 + 1) / 1000 = 1 kHz; all APB dividers are one. */
    htim3.Init.Prescaler = 7U;
    htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim3.Init.Period = BOARD_PWM_PERIOD - 1U;
    htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    if (HAL_TIM_PWM_Init(&htim3) != HAL_OK) return false;
    channel.OCMode = TIM_OCMODE_PWM1;
    channel.Pulse = 0U;
    channel.OCPolarity = TIM_OCPOLARITY_HIGH;
    channel.OCFastMode = TIM_OCFAST_DISABLE;
    if (HAL_TIM_PWM_ConfigChannel(&htim3, &channel, TIM_CHANNEL_1) != HAL_OK) return false;
    if (HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1) != HAL_OK) return false;
    pins.Pin = BOARD_PWM_PIN;
    pins.Mode = GPIO_MODE_AF_PP;
    pins.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOA, &pins);
    return true;
}

bool Board_Init(void)
{
    SafeOutputs();
    return Clock_Init() && Peripherals_Init();
}
bool Board_StartPressed(void) { return HAL_GPIO_ReadPin(GPIOB, BOARD_START_PIN) == GPIO_PIN_RESET; }
bool Board_StopPressed(void) { return HAL_GPIO_ReadPin(GPIOB, BOARD_STOP_PIN) == GPIO_PIN_RESET; }
void Board_SetLeds(bool green, bool red)
{
    HAL_GPIO_WritePin(GPIOB, BOARD_GREEN_PIN, green ? GPIO_PIN_SET : GPIO_PIN_RESET);
    HAL_GPIO_WritePin(GPIOB, BOARD_RED_PIN, red ? GPIO_PIN_SET : GPIO_PIN_RESET);
}
void Board_SetPwm(uint8_t percent)
{
    uint32_t compare = percent <= 100U ? (BOARD_PWM_PERIOD * percent) / 100U : 0U;
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, compare);
}
bool Board_I2cWrite(uint8_t address7, const uint8_t *data, uint16_t size)
{
    if (address7 > 0x7FU || data == NULL || size == 0U || size > 32U) return false;
    return HAL_I2C_Master_Transmit(&hi2c1, (uint16_t)(address7 << 1U),
                                  (uint8_t *)data, size, 5U) == HAL_OK;
}
_Noreturn void Board_Panic(void)
{
    __disable_irq();
    __HAL_RCC_TIM3_CLK_ENABLE();
    TIM3->CCER = 0U;
    TIM3->CR1 = 0U;
    SafeOutputs();
    for (;;) { __NOP(); }
}
