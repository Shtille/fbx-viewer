#version 330 core

uniform vec2 u_position;
uniform float u_aspect_ratio;

layout(location = 0) in vec4 a_position;

out vec2 v_texcoord;

void main()
{
	float kS2CX = 2.0/u_aspect_ratio;
	const float kS2CY = 2.0;
	vec2 screen_pos = u_position + a_position.xy;
	vec2 clip_pos;
	clip_pos.x = kS2CX * screen_pos.x - 1.0;
	clip_pos.y = kS2CY * screen_pos.y - 1.0;
    gl_Position = vec4(clip_pos, 0.0, 1.0);
    v_texcoord = a_position.zw;
}