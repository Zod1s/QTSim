use crate::utils::*;
use std::fs::{metadata, remove_file, rename, File};
use std::io::{BufRead, BufReader, LineWriter, Write};
use std::path::Path;
use std::process::Command;

pub fn plot_bloch_sphere(plot: &mut plotpy::Plot) -> SolverResult<()> {
    let mut sphere = plotpy::Surface::new();
    let spherecolor = "#00000020";
    let curvecolor = "#000000";
    let pointcolor = "#000000";
    let textcolor = "#000000";

    sphere
        .set_surf_color(spherecolor)
        .draw_sphere(&[0.0, 0.0, 0.0], 1.0, 40, 40)?;

    let mut xcurve = plotpy::Curve::new();
    xcurve
        .set_line_width(2.0)
        .set_line_color(curvecolor)
        .points_3d_begin()
        .points_3d_add(-1.0, 0.0, 0.0)
        .points_3d_add(1.0, 0.0, 0.0)
        .points_3d_end();

    let mut ycurve = plotpy::Curve::new();
    ycurve
        .set_line_width(2.0)
        .set_line_color(curvecolor)
        .points_3d_begin()
        .points_3d_add(0.0, -1.0, 0.0)
        .points_3d_add(0.0, 1.0, 0.0)
        .points_3d_end();

    let mut zcurve = plotpy::Curve::new();
    zcurve
        .set_line_width(2.0)
        .set_line_color(curvecolor)
        .points_3d_begin()
        .points_3d_add(0.0, 0.0, -1.0)
        .points_3d_add(0.0, 0.0, 1.0)
        .points_3d_end();

    let mut xsign = plotpy::Curve::new();
    xsign
        .set_line_width(5.0)
        .set_marker_style("o")
        .set_line_color(pointcolor)
        .points_3d_begin()
        .points_3d_add(1.0, 0.0, 0.0)
        .points_3d_end();

    let mut ysign = plotpy::Curve::new();
    ysign
        .set_line_width(5.0)
        .set_marker_style("o")
        .set_line_color(pointcolor)
        .points_3d_begin()
        .points_3d_add(0.0, 1.0, 0.0)
        .points_3d_end();

    let mut zsign = plotpy::Curve::new();
    zsign
        .set_line_width(5.0)
        .set_marker_style("o")
        .set_line_color(pointcolor)
        .points_3d_begin()
        .points_3d_add(0.0, 0.0, 1.0)
        .points_3d_end();

    let mut xtext = plotpy::Text::new();
    xtext
        .set_color(textcolor)
        .set_align_vertical("center")
        .set_align_horizontal("center")
        .set_fontsize(20.0)
        .set_bbox(false);
    xtext.draw_3d(1.0, 0.0, 0.1, "|+>");

    let mut ytext = plotpy::Text::new();
    ytext
        .set_color(textcolor)
        .set_align_vertical("center")
        .set_align_horizontal("center")
        .set_fontsize(20.0)
        .set_bbox(false);
    ytext.draw_3d(0.0, 1.0, 0.1, "|+i>");

    let mut ztext = plotpy::Text::new();
    ztext
        .set_color(textcolor)
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

pub fn constrainedlayout(path: &str, plot: &mut plotpy::Plot, show: bool) -> SolverResult<()> {
    // plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    plot.set_save_tight(true).save(&format!("{}.svg", path))?;
    let filename = &format!("{}.py", path);

    let file = File::open(filename)?;
    let lines = BufReader::new(file).lines();
    let tempfilename = &format!("{}temp.py", path);
    let tempfile = File::create(tempfilename)?;
    let mut linewrite = LineWriter::new(tempfile);

    let mut flag = 0;
    for line in lines.map_while(Result::ok) {
        if flag == 0 {
            flag = if line.starts_with("import") { 1 } else { 0 };
            linewrite.write(format!("{}\n", line).as_bytes())?;
        } else if flag == 1 {
            flag = if line.starts_with("import") { 1 } else { 2 };
            linewrite.write(format!("{}\n", line).as_bytes())?;
        } else {
            linewrite.write("plt.rcParams['figure.constrained_layout.use'] = True\n".as_bytes())?;
            flag = 0;
        }
    }

    if show {
        linewrite.write("plt.show()\n".as_bytes());
    }

    linewrite.flush()?;

    let res = match Command::new("python3").arg(tempfilename).output() {
        Ok(o) => Ok(()),
        Err(e) => {
            eprintln!("{}", e);
            Err(SolverError::IoError)
        }
    };

    if metadata(filename).is_ok() {
        remove_file(filename)?;
    }

    if metadata(tempfilename).is_ok() {
        remove_file(tempfilename)?;
    }

    res
}
