/*
 * Kin clip firmware: USB microphone + USB serial on an ESP32-S3.
 *
 * NOT COMPILED OR FLASHED. Written against ESP-IDF 5.x and the esp_tinyusb
 * component; expect small API adjustments on first build (see ../README.md).
 *
 *  - I2S RX reads the INMP441 (16 kHz, 32-bit frames, mono) and converts to 16-bit PCM.
 *  - TinyUSB exposes a UAC2 microphone ("Kin Clip") and a CDC serial port.
 *  - GPIO7 senses the mute switch; the switch itself cuts the microphone's power in hardware.
 *  - Lines from the host set the WS2812 status LED.
 */
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "tinyusb.h"
#include "tusb.h"
#include "led_strip.h"

#define PIN_I2S_SCK 4
#define PIN_I2S_WS 5
#define PIN_I2S_SD 6
#define PIN_MUTE_SENSE 7
#define PIN_RGB 48
#define SAMPLE_RATE 16000
#define FRAME_SAMPLES 160 /* 10 ms */

static const char *TAG = "kin-clip";
static i2s_chan_handle_t rx;
static led_strip_handle_t strip;

static void led(uint8_t r, uint8_t g, uint8_t b) {
    led_strip_set_pixel(strip, 0, r, g, b);
    led_strip_refresh(strip);
}

/* Host -> clip: "L<pattern>\n". Blinking and pulsing are drawn by led_task from the current pattern. */
static char pattern[24] = "green_solid";

static void serial_send(const char *line) {
    tud_cdc_write_str(line);
    tud_cdc_write_flush();
}

static void serial_task(void *arg) {
    char buf[48];
    int n = 0;
    bool last_muted = false;
    int64_t last_report = 0;
    bool greeted = false;
    for (;;) {
        if (tud_cdc_connected() && !greeted) {
            serial_send("HELLO kin-clip 1\n");
            greeted = true;
        }
        if (!tud_cdc_connected()) greeted = false;
        while (tud_cdc_available()) {
            char c;
            tud_cdc_read(&c, 1);
            if (c == '\n' || n >= (int)sizeof(buf) - 1) {
                buf[n] = 0;
                if (buf[0] == 'L') {
                    strncpy(pattern, buf + 1, sizeof(pattern) - 1);
                    pattern[sizeof(pattern) - 1] = 0;
                }
                n = 0;
            } else if (c != '\r') {
                buf[n++] = c;
            }
        }
        bool muted = gpio_get_level(PIN_MUTE_SENSE) == 0;
        int64_t now = esp_timer_get_time();
        if (tud_cdc_connected() && (muted != last_muted || now - last_report > 2000000)) {
            serial_send(muted ? "M1\n" : "M0\n");
            last_muted = muted;
            last_report = now;
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

static void led_task(void *arg) {
    uint32_t t = 0;
    for (;;) {
        bool on = (t / 5) % 2 == 0;                                   /* 500 ms half period at 100 ms ticks */
        uint8_t pulse = 20 + (uint8_t)(((t % 10) < 5 ? t % 10 : 10 - t % 10) * 20);
        if (!strcmp(pattern, "off")) led(0, 0, 0);
        else if (!strcmp(pattern, "green_solid")) led(0, 40, 0);
        else if (!strcmp(pattern, "green_pulse")) led(0, pulse, 0);
        else if (!strcmp(pattern, "blue_pulse")) led(0, 0, pulse);
        else if (!strcmp(pattern, "amber_blink")) led(on ? 60 : 0, on ? 30 : 0, 0);
        else if (!strcmp(pattern, "red_solid")) led(60, 0, 0);
        t++;
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

static void audio_task(void *arg) {
    static int32_t raw[FRAME_SAMPLES];
    static int16_t pcm[FRAME_SAMPLES];
    for (;;) {
        size_t got = 0;
        if (i2s_channel_read(rx, raw, sizeof(raw), &got, 100) != ESP_OK || got == 0) continue;
        int n = got / sizeof(int32_t);
        for (int i = 0; i < n; i++) pcm[i] = (int16_t)(raw[i] >> 14); /* INMP441: 24 bits left-aligned in 32 */
        /* When muted the microphone has no power, so this reads silence; the host stops listening anyway. */
        tud_audio_write(pcm, n * sizeof(int16_t));
    }
}

void app_main(void) {
    gpio_config_t sense = {.pin_bit_mask = 1ULL << PIN_MUTE_SENSE, .mode = GPIO_MODE_INPUT, .pull_up_en = 1};
    gpio_config(&sense);

    led_strip_config_t sc = {.strip_gpio_num = PIN_RGB, .max_leds = 1};
    led_strip_rmt_config_t rc = {.resolution_hz = 10 * 1000 * 1000};
    ESP_ERROR_CHECK(led_strip_new_rmt_device(&sc, &rc, &strip));

    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&cc, NULL, &rx));
    i2s_std_config_t std = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {.bclk = PIN_I2S_SCK, .ws = PIN_I2S_WS, .din = PIN_I2S_SD, .dout = I2S_GPIO_UNUSED,
                     .mclk = I2S_GPIO_UNUSED},
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx, &std));
    ESP_ERROR_CHECK(i2s_channel_enable(rx));

    /* USB descriptors: product string "Kin Clip", UAC2 microphone 16 kHz mono + CDC.
     * Set CONFIG_TINYUSB_DESC_PRODUCT_STRING="Kin Clip" in sdkconfig.defaults. */
    const tinyusb_config_t tusb = {0};
    ESP_ERROR_CHECK(tinyusb_driver_install(&tusb));

    xTaskCreate(audio_task, "audio", 4096, NULL, 6, NULL);
    xTaskCreate(serial_task, "serial", 3072, NULL, 3, NULL);
    xTaskCreate(led_task, "led", 2048, NULL, 2, NULL);
    ESP_LOGI(TAG, "Kin clip ready");
}
