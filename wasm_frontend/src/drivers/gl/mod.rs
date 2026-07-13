//! Shared WebGL2 infrastructure for every GPU driver.
//!
//! Before this module existed, the three GL drivers each carried private
//! copies of program linking, GPU timer queries, matrix math, shadow-map
//! target setup, and light selection. One definition of each now lives
//! here; the drivers keep only what is genuinely theirs (their resources
//! and their draw order).
//!
//! | module           | concern                                          |
//! |------------------|--------------------------------------------------|
//! | [`program`]      | context creation, shader compile/link            |
//! | [`timer`]        | `EXT_disjoint_timer_query_webgl2` frame timing   |
//! | [`math`]         | camera/ortho/look-at matrices                    |
//! | [`lights`]       | per-frame light selection + uniform packing      |
//! | [`shadow_target`]| depth-only FBO + texture for the hero light      |
//! | [`visibility`]   | conservative sphere visibility tests             |

pub mod light_volume;
pub mod lights;
pub mod math;
pub mod program;
pub mod shadow_target;
pub mod timer;
pub mod visibility;
