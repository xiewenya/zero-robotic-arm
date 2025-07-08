#include "robot.h"
#include "usart.h"
#include "string.h"
#include <stdio.h>
#include "robot_cmd.h"
#include "Emm_V5.h"

static int robot_soft_reset_handle(uint32_t joint_id, float *param);
void robot_mqtt_handle(struct robot_cmd *cmd)
{
	float joints_angle[ROBOT_MAX_JOINT_NUM] = {0};
	int strlen = 0;

	int result = sscanf(cmd->cmd, "+MQTTSUBRECV:0,\"arm/change\",%d,MCU %f %f %f %f %f %f", &strlen, &joints_angle[0], &joints_angle[1], &joints_angle[2],
			&joints_angle[3], &joints_angle[4], &joints_angle[5]);
	
	if (result < 7) { // 解析失败
		LOG("mqtt msg parse error: %s\n", cmd->cmd);
		return;
	}
	
	for (int i = 0; i < ROBOT_MAX_JOINT_NUM; i++) {
		robot_send_abs_rotate_event(i, joints_angle[i]);
	}
}

static int robot_remote_enable_handle(uint32_t joint_id, float *param)
{
	(void)joint_id;
	(void)param;
	robot_soft_reset_handle(joint_id, param);	/* 复位 */
	ROBOT_STATUS_SET(g_robot.status, ROBOT_RMODE_ENABLE);
	return robot_send_remote_event();
}

static int robot_remote_disable_handle(uint32_t joint_id, float *param)
{
	(void)joint_id;
	(void)param;
	ROBOT_STATUS_CLEAR(g_robot.status, ROBOT_RMODE_ENABLE);
	robot_soft_reset_handle(joint_id, param);	/* 复位 */
	return pdPASS;	
}

static int robot_rel_rotate_handle(uint32_t joint_id, float *param)
{
	return robot_send_rel_rotate_event(joint_id, param[0]);
}

static int robot_auto_handle(uint32_t joint_id, float *param)
{
	(void)joint_id;
	return robot_send_auto_event((struct position *)param);
}

static int robot_hard_reset_handle(uint32_t joint_id, float *param)
{
	(void)joint_id;
	(void)param;
	return robot_send_reset_event(true);	
}

static int robot_soft_reset_handle(uint32_t joint_id, float *param)
{
	(void)joint_id;
	(void)param;
	return robot_send_reset_event(false);	
}

static int robot_time_func_handle(uint32_t joint_id, float *param)
{
	(void)joint_id;
	return robot_send_time_func_event(param[0] * 1000);
}

static int robot_remote_event_handle(uint32_t joint_id, float *param)
{
	if (!ROBOT_STATUS_IS(g_robot.status, ROBOT_RMODE_ENABLE)) {
		return pdPASS;
	}

	float vx = -param[0] * ROBOT_REMOTE_MAX_VELOCITY;
	float vy = param[1] * ROBOT_REMOTE_MAX_VELOCITY;
	float vz = (param[4] - param[5]) / 2 * ROBOT_REMOTE_MAX_VELOCITY;
	float rx = -param[3] * ROBOT_REMOTE_MAX_RPM;
	float ry = param[2] * ROBOT_REMOTE_MAX_RPM;
	
	taskENTER_CRITICAL();
	g_remote_control.vx = vx;
	g_remote_control.vy = vy;
	g_remote_control.vz = vz;
	g_remote_control.rx = rx;
	g_remote_control.ry = ry;
	taskEXIT_CRITICAL();

	return pdPASS;
}

static int robot_zero_handle(uint32_t joint_id, float *param)
{
	(void)joint_id;
	(void)param;
	LOG("robot reset zero.\n");
	for (int i = 0; i < ROBOT_MAX_JOINT_NUM; i++) {
		Emm_V5_Reset_CurPos_To_Zero(joint_id + 1);
		vTaskDelay(10);
	}
	return pdPASS;
}

static struct robot_cmd_info robot_uart1_cmd_table[] = {
	{"remote_event", robot_remote_event_handle},
	{"remote_enable", robot_remote_enable_handle},
	{"remote_disable", robot_remote_disable_handle},
	{"rel_rotate", robot_rel_rotate_handle},
	{"auto", robot_auto_handle},
	{"hard_reset", robot_hard_reset_handle},
	{"soft_reset", robot_soft_reset_handle},
	{"zero", robot_zero_handle},
	{"time_func", robot_time_func_handle},
	{NULL, NULL},
};

void robot_uart1_handle(struct robot_cmd *rb_cmd)
{
	static uint32_t target_joint_id = 0;
	static char event_type[20] = {0};
	float param[6] = {0};
	char *cmd = rb_cmd->cmd;
	int ret;

	ret = sscanf(cmd, "%19s %f %f %f %f %f %f", event_type, &param[0], &param[1], &param[2], 
		&param[3], &param[4], &param[5]);
	if (ret < 1) { // 解析失败
        LOG("event_type parse error: %s\n", cmd);
        return;
    }

	if ((strcmp(event_type, "joint") == 0) && (param[0] > 0)) {
        target_joint_id = param[0] - 1;
        LOG("set target joint id: %u\n", target_joint_id);
        return;
    }

	for (int i = 0; robot_uart1_cmd_table[i].event_type != NULL; i++) {
		if (strcmp(event_type, robot_uart1_cmd_table[i].event_type) == 0) {
			ret = robot_uart1_cmd_table[i].cmd_func(target_joint_id, param);
			if (ret != pdPASS) {
				LOG("[ERROR] [jid:%d] event_type:%s param:%.2f %.2f %.2f\n", target_joint_id, event_type, param[0], param[1], param[2]);
				return;
			}
			return;
		}
	}

	LOG("uart cmd parse error: %s\n", cmd);
	return;
}

