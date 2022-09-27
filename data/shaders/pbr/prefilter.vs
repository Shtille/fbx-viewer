#version 330 core

layout(location = 0) in vec3 a_position;

uniform mat4 u_projection;
uniform mat4 u_view;

smooth out vec3 v_position;

void main()
{
	vec4 clip_position = vec4(a_position, 1.0);
	mat4 inverse_projection = inverse(u_projection);
	mat3 inverse_view = transpose(mat3(u_view));
	vec3 view_position = vec3(inverse_projection * clip_position);
	vec3 world_position = inverse_view * view_position;
	v_position = world_position;
    gl_Position = clip_position;
}