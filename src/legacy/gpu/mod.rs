use ocl::{Buffer, flags, ProQue, Queue};
use ocl::core::{Float3, Uint2};

use crate::{
  legacy::argmax::ArgmaxResult,
  geometry::{Point, TLBR, Circle},
  error::{ErrorKind::NoneError, Result}
};

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
  args: Args,
  image_size: Point<u32>
}

type Framebuffer = image::ImageBuffer<image::Luma<f32>, Vec<f32>>;
const WORKGROUP_SIZE: usize = 128;

impl KernelWrapper {

  pub fn load_source() -> std::io::Result<String> {
    std::fs::read_to_string("kernel/main.cl")
  }

  fn build_buffers(
    queue: Queue,
    framebuffer: &image::ImageBuffer<image::Luma<f32>, Vec<f32>>,
  ) -> Result<Args> {
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

  fn build_kernels(que: &ProQue, args: &Args, image_size: Point<u32>) -> Result<Kernels> {
    Ok(Kernels {
      main: que.kernel_builder("main")
        .arg(&args.framebuffer)
        .arg(image_size.x)
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
        .arg_named("image_size", Uint2::new(0, 0))
        .arg_named("coords_offset", Uint2::new(0, 0))
        .arg_named("circle", Float3::new(0.0, 0.0, 0.0))
        .build()?
    })
  }

  pub fn new(framebuffer: &Framebuffer) -> Result<KernelWrapper> {

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

    let image_size = Point { x: framebuffer.width(), y: framebuffer.height() };

    let kernels = Self::build_kernels(&main_que, &args, image_size)?;

    Ok(KernelWrapper { main_que, kernels, args, image_size })
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

  pub fn find_max(&mut self) -> Result<ArgmaxResult<f32>> {

    const ARGMAX_SIZE: usize = std::mem::size_of::<ArgmaxResult<u32>>();

    // phase 0
    let mut ret_len = self.args.reduced_result.len() / ARGMAX_SIZE;
    unsafe {
      self.kernels.main.enq()?;
    };

    // phase 1
    while ret_len / WORKGROUP_SIZE > 0 && ret_len % WORKGROUP_SIZE == 0  {
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

    Ok(result
      .into_iter()
      .map(|x| ArgmaxResult {
        point: (x.point.into(): Point<f32>) / (self.image_size.into(): Point<f32>),
        distance: x.distance
      })
      .max_by(|a, b| a.distance.total_cmp(&b.distance)).ok_or(NoneError)?
    )
  }

  pub fn write_to_device(&self, dist_map: &Framebuffer) -> ocl::Result<()> {
    self.args.framebuffer.write(dist_map.as_raw()).enq()
  }

  pub fn read_from_device(&self, dist_map: &mut Framebuffer) -> ocl::Result<()> {
    self.args.framebuffer.read(dist_map.as_mut()).enq()
  }

  pub fn insert_sdf_circle(&mut self, circle: crate::geometry::Circle) -> Result<()> {
    let domain = TLBR {
      tl: Point { x: 0.0, y: 0.0 },
      br: Point { x: 1.0, y: 1.0 }
    };
    self.insert_sdf_circle_domain(circle, domain)
  }

  pub fn insert_sdf_circle_domain(&mut self, circle: Circle, domain: TLBR<f32>) -> Result<()> {
    let domain = TLBR {
      tl: (Point {
        x: domain.tl.x.max(0.0),
        y: domain.tl.y.max(0.0)
      } * (self.image_size.into(): Point<f32>)).into(): Point<u32>,
      br: (Point {
        x: domain.br.x.min(1.0),
        y: domain.br.y.min(1.0)
      } * (self.image_size.into(): Point<f32>)).into(): Point<u32>
    };

    self.kernels.insert_sdf_circle.set_arg(
      "image_size",
      Uint2::new(self.image_size.x, domain.br.x - domain.tl.x)
    )?;
    self.kernels.insert_sdf_circle.set_arg(
      "coords_offset",
      Uint2::new(domain.tl.x, domain.tl.y)
    )?;
    self.kernels.insert_sdf_circle.set_arg(
      "circle",
      Float3::new(circle.xy.x, circle.xy.y, circle.r)
    )?;
    self.kernels.insert_sdf_circle.set_default_global_work_size(
      ((domain.br.x - domain.tl.x) * (domain.br.y - domain.tl.y)).into()
    );
    unsafe {
      self.kernels.insert_sdf_circle.enq()?;
    }
    Ok(())
  }
}
