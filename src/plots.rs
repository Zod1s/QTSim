use crate::utils::SolverResult;
use nalgebra as na;
use plotpy;

pub fn plot_bloch_sphere(plot: &mut plotpy::Plot) -> SolverResult<()> {
    let mut sphere = plotpy::Surface::new();
    sphere
        .set_surf_color("#00000020")
        .draw_sphere(&[0.0, 0.0, 0.0], 1.0, 40, 40)?;

    let mut xcurve = plotpy::Curve::new();
    xcurve
        .set_line_width(2.0)
        .set_line_color("#000000")
        .points_3d_begin()
        .points_3d_add(-1.0, 0.0, 0.0)
        .points_3d_add(1.0, 0.0, 0.0)
        .points_3d_end();

    let mut ycurve = plotpy::Curve::new();
    ycurve
        .set_line_width(2.0)
        .set_line_color("#000000")
        .points_3d_begin()
        .points_3d_add(0.0, -1.0, 0.0)
        .points_3d_add(0.0, 1.0, 0.0)
        .points_3d_end();

    let mut zcurve = plotpy::Curve::new();
    zcurve
        .set_line_width(2.0)
        .set_line_color("#000000")
        .points_3d_begin()
        .points_3d_add(0.0, 0.0, -1.0)
        .points_3d_add(0.0, 0.0, 1.0)
        .points_3d_end();

    let mut xsign = plotpy::Curve::new();
    xsign
        .set_line_width(5.0)
        .set_marker_style("o")
        .set_line_color("#0000ff")
        .points_3d_begin()
        .points_3d_add(1.0, 0.0, 0.0)
        .points_3d_end();

    let mut ysign = plotpy::Curve::new();
    ysign
        .set_line_width(5.0)
        .set_marker_style("o")
        .set_line_color("#0000ff")
        .points_3d_begin()
        .points_3d_add(0.0, 1.0, 0.0)
        .points_3d_end();

    let mut zsign = plotpy::Curve::new();
    zsign
        .set_line_width(5.0)
        .set_marker_style("o")
        .set_line_color("#0000ff")
        .points_3d_begin()
        .points_3d_add(0.0, 0.0, 1.0)
        .points_3d_end();

    let mut xtext = plotpy::Text::new();
    xtext
        .set_color("#0000ff")
        .set_align_vertical("center")
        .set_align_horizontal("center")
        .set_fontsize(20.0)
        .set_bbox(false);
    xtext.draw_3d(1.0, 0.0, 0.1, "|+>");

    let mut ytext = plotpy::Text::new();
    ytext
        .set_color("#0000ff")
        .set_align_vertical("center")
        .set_align_horizontal("center")
        .set_fontsize(20.0)
        .set_bbox(false);
    ytext.draw_3d(0.0, 1.0, 0.1, "|+i>");

    let mut ztext = plotpy::Text::new();
    ztext
        .set_color("#0000ff")
        .set_align_vertical("center")
        .set_align_horizontal("center")
        .set_fontsize(20.0)
        .set_bbox(false);
    ztext.draw_3d(0.0, 0.0, 1.1, "|0>");

    plot.add(&sphere)
        .add(&xcurve)
        .add(&xsign)
        .add(&xtext)
        .add(&ycurve)
        .add(&ysign)
        .add(&ytext)
        .add(&zcurve)
        .add(&zsign)
        .add(&ztext);

    Ok(())
}
