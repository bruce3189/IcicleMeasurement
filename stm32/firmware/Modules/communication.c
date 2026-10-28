#include "communication.h"
#include "board.h"
#include "protocol.h"
#include <string.h>

#define RX_CAPACITY 256U
#define POLL_BUDGET 64U
static uint8_t rx_byte, rx_ring[RX_CAPACITY], tx_frame[64];
static volatile uint16_t rx_head, rx_tail;
static volatile bool stream_error, restart_rx;
volatile CommunicationStats communication_stats;

static uint32_t Lock(void) { uint32_t mask = __get_PRIMASK(); __disable_irq(); return mask; }
static void Unlock(uint32_t mask) { __set_PRIMASK(mask); }

bool Communication_Init(void)
{
    rx_head = rx_tail = 0U;
    stream_error = restart_rx = false;
    communication_stats = (CommunicationStats){0};
    return HAL_UART_Receive_IT(&huart1, &rx_byte, 1U) == HAL_OK;
}

void Communication_ResetRx(void)
{
    uint32_t mask = Lock();
    (void)HAL_UART_AbortReceive(&huart1); /* No DMA, no wait for bytes. */
    __HAL_UART_CLEAR_OREFLAG(&huart1);
    rx_head = rx_tail = 0U;
    stream_error = false;
    restart_rx = HAL_UART_Receive_IT(&huart1, &rx_byte, 1U) != HAL_OK;
    if (restart_rx) { ++communication_stats.restarts_failed; stream_error = true; }
    Unlock(mask);
    Protocol_Reset();
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *uart)
{
    if (uart != &huart1) return;
    if (uart->ErrorCode != HAL_UART_ERROR_NONE) {
        stream_error = restart_rx = true;
        return; /* Do not rearm here: HAL still has to finish its error path. */
    }
    uint16_t next = (uint16_t)((rx_head + 1U) % RX_CAPACITY);
    if (next == rx_tail) {
        ++communication_stats.rx_overflows;
        stream_error = true;
    } else {
        rx_ring[rx_head] = rx_byte;
        rx_head = next;
    }
    if (HAL_UART_Receive_IT(uart, &rx_byte, 1U) != HAL_OK)
        stream_error = restart_rx = true;
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *uart)
{
    if (uart == &huart1) {
        ++communication_stats.uart_errors;
        stream_error = restart_rx = true;
    }
}

void Communication_Poll(AppControl *app, uint32_t now_ms)
{
    for (unsigned count = 0U; count < POLL_BUDGET; ++count) {
        uint8_t byte;
        uint32_t mask = Lock();
        if (stream_error || restart_rx) {
            Unlock(mask);
            Communication_ResetRx();
            App_TransportLost(app);
            return;
        }
        if (rx_head == rx_tail) { Unlock(mask); return; }
        byte = rx_ring[rx_tail];
        rx_tail = (uint16_t)((rx_tail + 1U) % RX_CAPACITY);
        Unlock(mask);
        Protocol_Feed(byte, app, now_ms);
    }
}

bool Communication_WriteFrame(const uint8_t *data, uint16_t size)
{
    if (data == NULL || size == 0U || size > sizeof(tx_frame) ||
        huart1.gState != HAL_UART_STATE_READY) return false;
    memcpy(tx_frame, data, size);
    return HAL_UART_Transmit_IT(&huart1, tx_frame, size) == HAL_OK;
}
