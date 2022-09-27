#version 330 core

layout(location = 0) in vec3 a_position;

out vec2 v_texcoord;

void main()
{
    vec4 clip_position = vec4(a_position, 1.0);
    v_texcoord = (clip_position.xy + 1.0) * 0.5;
    gl_Position = clip_position;
}