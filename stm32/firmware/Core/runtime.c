/* CMSIS Reset_Handler calls newlib's constructor dispatcher. No filesystem,
 * printf, heap allocation, semihosting or serial logging is used here. */
void _init(void) {}
void _fini(void) {}
