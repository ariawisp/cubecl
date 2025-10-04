fn main() {
    #[cfg(feature = "metal4")]
    {
        use cubecl::metal4::Metal4Runtime;
        metal4_add::launch::<Metal4Runtime>(&Default::default());
    }
}

