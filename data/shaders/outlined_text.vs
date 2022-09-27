#version 330 core

layout(location = 0) in vec4 a_position;

out vec2 v_texcoord;

void main()
{
    gl_Position = vec4(a_position.xy, 0.0, 1.0);
    v_texcoord = a_position.zw;
}