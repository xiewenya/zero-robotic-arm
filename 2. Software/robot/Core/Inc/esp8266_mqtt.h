#ifndef ESP8266_MQTT_H
#define ESP8266_MQTT_H

#include "stm32f4xx_hal.h"

// 定义 WiFi 信息
#define WIFI_SSID       "小谢与小李的家"
#define WIFI_PASSWORD   "XGljh246810"

// 定义 MQTT 信息
#define MQTT_SERVER     "41wm32327tp1.vicp.fun"
#define MQTT_PORT       50802
#define MQTT_CLIENT_ID  "xiegeng_nb"
#define MQTT_TOPIC      "arm/change"
#define MQTT_USERNAME   "admin0"
#define MQTT_PASSWORD   "123456"

// 发送 AT 指令
void esp8266_send_at_command(const char *command);
// 等待 AT 指令响应
int esp8266_wait_response(const char *response, uint32_t timeout);
// ESP8266 连接 WiFi
int esp8266_connect_wifi(const char *ssid, const char *password);
// ESP8266 连接 MQTT 服务器
int esp8266_connect_mqtt(const char *server, uint16_t port, const char *client_id, const char *username, const char *password);
// ESP8266 订阅 MQTT 主题
int esp8266_subscribe_topic(const char *topic, uint8_t qos);
// ESP8266 发布 MQTT 消息
int esp8266_publish_message(const char *topic, const char *message, uint8_t qos);
// 初始化 ESP8266
int esp8266_mqtt_init(void);

#endif
