fn main() {
    #[cfg(feature = "metal4")]
    {
        use cubecl::metal4::Metal4Runtime;
        println!("-- Metal4 --");
        metal4_ops::run::<Metal4Runtime>(&Default::default());
    }
    #[cfg(feature = "wgpu")]
    {
        use cubecl::wgpu::WgpuRuntime;
        println!("-- WGPU --");
        metal4_ops::run::<WgpuRuntime>(&Default::default());
    }
}
