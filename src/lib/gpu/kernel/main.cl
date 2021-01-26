struct ArgmaxResult {
  float distance;
  uint x;
  uint y;
};

__kernel void main(
  __global float * framebuffer,
  __private unsigned int image_w,
  __global struct ArgmaxResult * const result
) {
  unsigned int global_id = get_global_id(0);
  unsigned int local_id = get_local_id(0);
  unsigned int workgroup_id = get_group_id(0);
  unsigned int local_size = get_local_size(0);
  unsigned int image_x = global_id % image_w;
  unsigned int image_y = global_id / image_w;
  __local struct ArgmaxResult fast_mem[128];

  fast_mem[local_id] = (struct ArgmaxResult){
    .distance = framebuffer[global_id],
    .x = image_x,
    .y = image_y
  };

  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

  for(int i = local_size / 2; i>=1; i/=2) {
    if (local_id < i) {
      if(fast_mem[local_id].distance < fast_mem[local_id + i].distance)
        fast_mem[local_id] = fast_mem[local_id + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0)
    result[workgroup_id] = fast_mem[0];
}

__kernel void find_max_phase1(
  __global struct ArgmaxResult * const reduced_result
) {
  unsigned int global_id = get_global_id(0);
  unsigned int local_id = get_local_id(0);
  unsigned int workgroup_id = get_group_id(0);
  unsigned int local_size = get_local_size(0);
  __local struct ArgmaxResult fast_mem[128];

  fast_mem[local_id] = reduced_result[global_id];

  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

  for(int i = local_size / 2; i>=1; i/=2) {
    if (local_id < i) {
      if(fast_mem[local_id].distance < fast_mem[local_id + i].distance)
        fast_mem[local_id] = fast_mem[local_id + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0)
    reduced_result[workgroup_id] = fast_mem[0];
}


//####### SDF
struct Circle {
  float2 xy;
  float r;
};

float sdf_circle(struct Circle circle, float2 pixel) {
  return length(pixel - circle.xy) - circle.r;
}

__kernel void insert_sdf_circle(
  __global float * framebuffer,
  __private unsigned int image_w,
  __private struct Circle circle
) {
  unsigned int global_id = get_global_id(0);
  unsigned int image_x = global_id % image_w;
  unsigned int image_y = global_id / image_w;
  float2 pixel = (float2)((float)image_x / (float)image_w, (float)image_y / (float)image_w); //?

  framebuffer[global_id] = min(framebuffer[global_id], sdf_circle(circle, pixel));
}