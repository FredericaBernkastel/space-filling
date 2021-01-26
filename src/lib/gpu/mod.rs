use ocl::{ProQue, Buffer, flags, Queue};
use crate::argmax::ArgmaxResult;
use ocl::core::Float3;

struct Kernels {
  main: ocl::Kernel,
  find_max_phase1: ocl::Kernel,
  insert_sdf_circle: ocl::Kernel
}

struct Args {
  framebuffer: Buffer<f32>,
  reduced_result: Buffer<u8>
}

pub struct KernelWrapper{
  main_que: ProQue,
  kernels: Kernels,
  args: Args
}

type Framebuffer = image::ImageBuffer<image::Luma<f32>, Vec<f32>>;
const WORKGROUP_SIZE: usize = 512;

impl KernelWrapper {

  pub fn load_source() -> std::io::Result<String> {
    std::fs::read_to_string("kernel/main.cl")
  }

  fn build_buffers(
    queue: Queue,
    framebuffer: &image::ImageBuffer<image::Luma<f32>, Vec<f32>>,
  ) -> ocl::Result<Args> {
    const RESULT_BPP: usize = std::mem::size_of::<ArgmaxResult<u32>>();
    let result_len = framebuffer.len() / WORKGROUP_SIZE;
    Ok(Args {
      framebuffer: Buffer::<f32>::builder()
        .len(framebuffer.len())
        .flags(flags::MEM_READ_ONLY)
        //.copy_host_slice(framebuffer)
        .queue(queue.clone())
        .build()?,
      reduced_result: Buffer::<u8>::builder()
        .len(result_len * RESULT_BPP)
        .queue(queue.clone())
        .build()?
    })
  }

  fn build_kernels(que: &ProQue, args: &Args, image_width: u32) -> ocl::Result<Kernels> {
    Ok(Kernels {
      main: que.kernel_builder("main")
        .arg(&args.framebuffer)
        .arg(image_width)
        .arg(&args.reduced_result)
        .global_work_size(args.framebuffer.len())
        .local_work_size(WORKGROUP_SIZE)
        .build()?,
      find_max_phase1: que.kernel_builder("find_max_phase1")
        .arg(&args.reduced_result)
        .local_work_size(WORKGROUP_SIZE)
        .build()?,
      insert_sdf_circle: que.kernel_builder("insert_sdf_circle")
        .arg(&args.framebuffer)
        .arg(image_width)
        .arg_named("circle", Float3::new(0.0, 0.0, 0.0))
        .global_work_size(args.framebuffer.len())
        .build()?
    })
  }

  pub fn new(framebuffer: &Framebuffer) -> ocl::Result<KernelWrapper> {

    let device = ocl::Device::list(
      ocl::Platform::default(), Some(ocl::flags::DEVICE_TYPE_GPU))?
      .first()
      .expect("No GPU devices found")
      .clone();

    //println!("opencl::device::info: {}", device.to_string());

    let main_que = ProQue::builder()
      .src(Self::load_source()?)
      .device(device)
      .build()?;

    let args = Self::build_buffers(
      main_que.queue().clone(),
      framebuffer,
    )?;

    let kernels = Self::build_kernels(&main_que, &args, framebuffer.width())?;

    Ok(KernelWrapper { main_que, kernels, args })
  }

  /*pub fn recompile(&mut self) -> ocl::Result<()>{

    /* Update strategy:
     * 1. compile new Program, migrate Device and Context, build Queue
     * 2. migrate device buffers into new queue
     * 3. rebuild kernels
     * 4. update kernel, program, device, context, and queue references
     */

    let que = ProQue::builder()
      .src(Self::load_source()?)
      .device(self.main_que.device())
      .context(self.main_que.context().clone())
      .build()?;

    self.args.framebuffer.set_default_queue(que.queue().clone());
    self.args.reduced_result.set_default_queue(que.queue().clone());

    self.kernels = Self::build_kernels(&que, &self.args)?;
    self.main_que = que;

    Ok(())
  }*/

  pub fn find_max(&mut self) -> ocl::Result<Vec<ArgmaxResult<u32>>> {
    const ARGMAX_SIZE: usize = std::mem::size_of::<ArgmaxResult<u32>>();

    // phase 0
    let mut ret_len = self.args.reduced_result.len() / ARGMAX_SIZE;
    unsafe {
      self.kernels.main.enq()?;
    };

    // phase 1
    if ret_len / WORKGROUP_SIZE > 0 && ret_len % WORKGROUP_SIZE == 0  {
      self.kernels.find_max_phase1.set_default_global_work_size(ret_len.into());
      ret_len = ret_len / WORKGROUP_SIZE;
      unsafe {
        self.kernels.find_max_phase1.enq()?;
      }
    }

    // read result
    let mut result = vec![ArgmaxResult::<u32>::default(); ret_len];
    unsafe {
      self.args.reduced_result.read(
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, ret_len * ARGMAX_SIZE)
      ).enq()?;
    }
    Ok(result)
  }

  pub fn write_to_device(&self, dist_map: &Framebuffer) -> ocl::Result<()> {
    self.args.framebuffer.write(dist_map.as_raw()).enq()
  }

  pub fn read_from_device(&self, dist_map: &mut Framebuffer) -> ocl::Result<()> {
    self.args.framebuffer.read(dist_map.as_mut()).enq()
  }

  pub fn insert_sdf_circle(&self, circle: crate::geometry::Circle) -> ocl::Result<()> {
    self.kernels.insert_sdf_circle.set_arg(
      "circle",
      Float3::new(circle.xy.x, circle.xy.y, circle.r)
    )?;
    unsafe {
      self.kernels.insert_sdf_circle.enq()?;
    }
    Ok(())
  }
}
