/* STM32F103C8T6: 64 KiB Flash, 20 KiB RAM; no dynamic heap. */
ENTRY(Reset_Handler)
MEMORY {
    FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 64K
    RAM (rw)   : ORIGIN = 0x20000000, LENGTH = 20K
}
PHDRS { flash PT_LOAD FLAGS(5); ram PT_LOAD FLAGS(6); }
_estack = ORIGIN(RAM) + LENGTH(RAM);
_Min_Stack_Size = 0x600;
SECTIONS {
    .isr_vector : { . = ALIGN(4); KEEP(*(.isr_vector)) } > FLASH :flash
    .text : {
        . = ALIGN(4);
        *(.text*) *(.rodata*) *(.glue_7*)
        KEEP(*(.init)) KEEP(*(.fini))
        . = ALIGN(4);
        _etext = .;
    } > FLASH :flash
    .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > FLASH :flash
    .ARM.exidx : {
        __exidx_start = .; *(.ARM.exidx*) __exidx_end = .;
    } > FLASH :flash
    .preinit_array : {
        PROVIDE_HIDDEN(__preinit_array_start = .);
        KEEP(*(.preinit_array*))
        PROVIDE_HIDDEN(__preinit_array_end = .);
    } > FLASH :flash
    .init_array : {
        PROVIDE_HIDDEN(__init_array_start = .);
        KEEP(*(SORT(.init_array.*))) KEEP(*(.init_array))
        PROVIDE_HIDDEN(__init_array_end = .);
    } > FLASH :flash
    .fini_array : {
        PROVIDE_HIDDEN(__fini_array_start = .);
        KEEP(*(SORT(.fini_array.*))) KEEP(*(.fini_array))
        PROVIDE_HIDDEN(__fini_array_end = .);
    } > FLASH :flash
    _sidata = LOADADDR(.data);
    .data : {
        . = ALIGN(4); _sdata = .;
        *(.data*) . = ALIGN(4); _edata = .;
    } > RAM AT> FLASH :ram
    .bss (NOLOAD) : {
        . = ALIGN(4); _sbss = .; __bss_start__ = .;
        *(.bss*) *(COMMON)
        . = ALIGN(4); _ebss = .; __bss_end__ = .;
    } > RAM :ram
    .stack_reserve (NOLOAD) : {
        . = ALIGN(8); . += _Min_Stack_Size; . = ALIGN(8);
    } > RAM :NONE
    .ARM.attributes 0 : { *(.ARM.attributes) }
    /DISCARD/ : { *(.note.GNU-stack) *(.comment) }
}
ASSERT(_ebss + _Min_Stack_Size <= _estack, "RAM data overlaps reserved stack")
