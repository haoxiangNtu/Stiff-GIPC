#pragma once

enum class BodyBoundaryType : int
{
    Free     = 0,
    Fixed    = 1,
    Motor    = 2,
    Animated = 3   // Soft translation drive: uses body_motor_params[0:2] as velocity, [3] as strength
};