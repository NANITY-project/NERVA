/* nebula_capi.h -- flat C interface to the Nebula IK engine, for host apps
 * (Blender add-on, Python studio server, game engines). All vectors are
 * float[3], quaternions float[4] as (w, x, y, z). The engine's coordinate
 * convention is Y-up, bones along local +X; hosts convert at the boundary.
 * Return values: >=0 / 1 = ok, -1 / 0 = failure, per function. */
#ifndef NEBULA_CAPI_H
#define NEBULA_CAPI_H
#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define NEBULA_API __declspec(dllexport)
#else
#  define NEBULA_API __attribute__((visibility("default")))
#endif

typedef struct NebulaEngine NebulaEngine;

enum { NEBULA_SOLVER_TWO_BONE = 0, NEBULA_SOLVER_FABRIK = 1 };

/* version 2 = adds the nebula_body_* functions (everything else is unchanged) */
NEBULA_API int           nebula_abi_version(void);
NEBULA_API NebulaEngine* nebula_create(void);
NEBULA_API void          nebula_destroy(NebulaEngine*);

/* Returns chain id, or -1 (bad spec / duplicate name / too many joints).
 * root_quat may be NULL (identity). */
NEBULA_API int nebula_add_chain(NebulaEngine*, const char* name, const float root[3],
                                const float* root_quat, int solver, int n_joints,
                                const float* bone_lengths, const float pole[3],
                                float torso_lean_m, float max_speed_m_s);
NEBULA_API int nebula_find_chain(NebulaEngine*, const char* name);
NEBULA_API int nebula_chain_count(NebulaEngine*);
NEBULA_API int nebula_joint_count(NebulaEngine*, int chain);

NEBULA_API int nebula_set_anchor(NebulaEngine*, int chain, const float pos[3], const float* quat);
NEBULA_API int nebula_teleport(NebulaEngine*, int chain, const float hand[3]);
NEBULA_API int nebula_set_pole(NebulaEngine*, int chain, const float dir[3]);
NEBULA_API int nebula_set_rest(NebulaEngine*, int chain, const float hand[3]);
NEBULA_API int nebula_set_elbow_limits(NebulaEngine*, const char* chain, float min_rad, float max_rad);
NEBULA_API int nebula_set_max_bend(NebulaEngine*, const char* chain, int joint, float rad);

/* Text protocol: REACH / POINT / ORIENT / RESET (see README). 1 ok, 0 rejected. */
NEBULA_API int nebula_command(NebulaEngine*, const char* line);
NEBULA_API int nebula_reach(NebulaEngine*, int chain, const float target[3]);

/* Named tunable (MotionParams field): reaction_delay, fitts_a, fitts_b,
 * fitts_width, min_duration, max_duration, path_curvature, pole_relax_rate,
 * lean_omega, idle_amp_m, breath_amp_m, breath_hz, solve_epsilon_m. */
NEBULA_API int nebula_set_param(NebulaEngine*, const char* key, float value);

NEBULA_API void nebula_update(NebulaEngine*, float dt);

/* out: 3*(n_joints+1) floats (root, each joint, end effector). Returns n_joints+1. */
NEBULA_API int nebula_world_positions(NebulaEngine*, int chain, float* out);
/* out: 4*n_joints floats, local rotations (w,x,y,z). Returns n_joints. */
NEBULA_API int nebula_local_rotations(NebulaEngine*, int chain, float* out);
NEBULA_API int nebula_is_settled(NebulaEngine*, int chain);
NEBULA_API int nebula_hand_velocity(NebulaEngine*, int chain, float out[3]);

/* ===================== full-body humanoid (ik_body.hpp) =====================
 * Coordinates are world metres, Y up; at yaw 0 the body faces +Z and its left is +X.
 * Commands are text lines (HAND / FOOT / PELVIS / FACE / LOOK / TORSO / GRIP / STAND / BALANCE / SURFACE);
 * a refusal always carries a reason in `msg`. */
typedef struct NebulaBody NebulaBody;

/* scale 1.0 = ~1.75 m. finger_lod: 0 none, 1 one bone per finger, 2 full phalanges. */
NEBULA_API NebulaBody* nebula_body_create(float scale, int finger_lod, int balance_assist);
NEBULA_API void        nebula_body_destroy(NebulaBody*);
/* 1 = accepted, 0 = refused. msg (optional) receives the reply, NUL-terminated and truncated to msg_cap. */
NEBULA_API int  nebula_body_command(NebulaBody*, const char* line, char* msg, int msg_cap);
NEBULA_API void nebula_body_update(NebulaBody*, float dt);
/* keys: safe_margin, step_time_scale, max_stride, max_step_up, max_step_down, fall_margin, fall_time,
 *       shift_settle_speed, hand_idle_m, balance (1 assist / 0 none) */
NEBULA_API int  nebula_body_set_param(NebulaBody*, const char* key, float value);
NEBULA_API int  nebula_body_add_surface(NebulaBody*, float x0, float x1, float z0, float z1, float y);
NEBULA_API void nebula_body_clear_surfaces(NebulaBody*);
NEBULA_API int  nebula_body_bone_count(NebulaBody*);
NEBULA_API const char* nebula_body_bone_name(NebulaBody*, int i);   /* valid until destroy; NULL if out of range */
NEBULA_API int  nebula_body_bone_parent(NebulaBody*, int i);        /* -1 for the root */
/* out: 10 floats per bone (head xyz, tail xyz, orientation w x y z). Returns the bone count. */
NEBULA_API int  nebula_body_bones(NebulaBody*, float* out);
/* JSON state + the feedback events since the last call. Returns the length written, or the length needed if cap was too small. */
NEBULA_API int  nebula_body_status_json(NebulaBody*, char* buf, int cap);

#ifdef __cplusplus
}
#endif
#endif
