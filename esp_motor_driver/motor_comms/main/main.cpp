/* BSD Socket API Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
#include "addr_from_stdin.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "lwip/err.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"
#include "protocol_examples_common.h"
#include <string.h>
#include <sys/param.h>

#include "motor_driver.hpp"
#include <string>

#ifndef CONFIG_EXAMPLE_IPV4
#error "This component requires IPV4"
#endif

#define HOST_IP_ADDR CONFIG_EXAMPLE_IPV4_ADDR
#define PORT CONFIG_EXAMPLE_PORT

static const char *TAG = "esp_motor_driver";

static auto motor = MotorDriver(MotorDriver::Config{.ENABLE_L = (gpio_num_t)15,
                                                    .ENABLE_R = (gpio_num_t)2,
                                                    .IN1 = (gpio_num_t)0,
                                                    .IN2 = (gpio_num_t)4,
                                                    .IN3 = (gpio_num_t)16,
                                                    .IN4 = (gpio_num_t)17});

static void tcp_client_task(void *pvParameters) {
  char host_ip[] = HOST_IP_ADDR;
  int addr_family = 0;
  int ip_protocol = 0;

  while (true) {
    struct sockaddr_in dest_addr;
    dest_addr.sin_addr.s_addr = inet_addr(host_ip);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(PORT);
    addr_family = AF_INET;
    ip_protocol = IPPROTO_IP;
    int sock = socket(addr_family, SOCK_STREAM, ip_protocol);
    if (sock < 0) {
      ESP_LOGE(TAG, "Unable to create socket: errno %d", errno);
      break;
    }
    ESP_LOGI(TAG, "Socket created, connecting to %s:%d", host_ip, PORT);

    int err = connect(sock, (struct sockaddr *)&dest_addr,
                      sizeof(struct sockaddr_in));
    if (err != 0) {
      ESP_LOGE(TAG, "Socket unable to connect: errno %d", errno);
      break;
    }
    ESP_LOGI(TAG, "Successfully connected");

    while (1) {
      char rx_buffer[128];

      std::string payload = "RQ";
      int err = send(sock, payload.c_str(), strlen(payload.c_str()), 0);
      if (err < 0) {
        ESP_LOGE(TAG, "Error occurred during sending: errno %d", errno);
        break;
      }

      // Receive motor data
      int len = recv(sock, rx_buffer, sizeof(rx_buffer) - 1, 0);
      if (len < 0) {
        ESP_LOGE(TAG, "recv failed: errno %d", errno);
        break;
      } else {
        rx_buffer[len] =
            0; // Null-terminate whatever we received and treat like a string
        ESP_LOGI(TAG, "Received %d bytes from %s:", len, host_ip);
        ESP_LOGI(TAG, "%s", rx_buffer);
      }

      // Parse data
      std::string message{rx_buffer};
      std::size_t found = message.find(" ");
      if (found != std::string::npos) {
        int dp_l = std::stoi(message.substr(0, found));
        int dp_r = std::stoi(message.substr(found + 1));

        ESP_LOGI(TAG, "Received %s", message.c_str());
        ESP_LOGI(TAG, "DP L R %d %d", dp_l, dp_r);

        // Send to motor drivers
        motor.drivelr(dp_l, dp_r);
      }

      vTaskDelay(2000 / portTICK_PERIOD_MS);
    }

    if (sock != -1) {
      ESP_LOGE(TAG, "Shutting down socket and restarting...");
      shutdown(sock, 0);
      close(sock);
    }
  }
  vTaskDelete(NULL);
}

void setup_wifi_client() {

  ESP_ERROR_CHECK(nvs_flash_init());
  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());

  /* This helper function configures Wi-Fi or Ethernet, as selected in
   * menuconfig. Read "Establishing Wi-Fi or Ethernet Connection" section in
   * examples/protocols/README.md for more information about this function.
   */
  ESP_ERROR_CHECK(example_connect());

  xTaskCreate(tcp_client_task, "tcp_client", 4096, NULL, 5, NULL);
}

extern "C" void app_main(void) {
  // Set up Wifi client connection
  setup_wifi_client();
}
