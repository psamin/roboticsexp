"""RTX render settings and frame capture through a Replicator rgb annotator (works headless on DGX Spark)."""
import numpy as np


def set_quality(mode: str = "pt", spp_total: int = 512, spp_frame: int = 16, denoise: bool = True):
    import carb
    s = carb.settings.get_settings()
    if mode == "pt":
        s.set("/rtx/rendermode", "PathTracing")
        s.set("/rtx/pathtracing/spp", spp_frame)
        s.set("/rtx/pathtracing/totalSpp", spp_total)
        s.set("/rtx/pathtracing/maxBounces", 6)
        s.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 6)
        s.set("/rtx/pathtracing/optixDenoiser/enabled", denoise)
        s.set("/rtx/pathtracing/adaptiveSampling/enabled", False)
    else:
        s.set("/rtx/rendermode", "RaytracedLighting")
        s.set("/rtx/directLighting/sampledLighting/enabled", True)
        s.set("/rtx/reflections/enabled", True)
        s.set("/rtx/indirectDiffuse/enabled", True)
        s.set("/rtx/ambientOcclusion/enabled", True)
    s.set("/rtx/post/histogram/enabled", True)          # auto exposure
    s.set("/rtx/post/histogram/whiteScale", 7.0)
    s.set("/rtx/post/tonemap/filmIso", 100.0)
    return {k: s.get(k) for k in ("/rtx/rendermode", "/rtx/pathtracing/totalSpp", "/rtx/post/histogram/enabled")}


class Grabber:
    """One render product + rgb annotator per camera."""

    def __init__(self, cams: dict, width: int, height: int):
        import omni.replicator.core as rep
        self.annot = {}
        self.products = {}
        for name, path in cams.items():
            rp = rep.create.render_product(path, (width, height))
            a = rep.AnnotatorRegistry.get_annotator("rgb")
            a.attach([rp])
            self.annot[name], self.products[name] = a, rp

    def get(self, name):
        d = self.annot[name].get_data()
        if d is None or getattr(d, "size", 0) == 0:
            return None
        return np.asarray(d)[..., :3].copy()


def image_stats(img: np.ndarray) -> dict:
    g = img.astype(np.float32).mean(axis=-1)
    return {"shape": list(img.shape), "mean": float(img.mean()), "std": float(img.std()),
            "dark_frac": float((g < 8).mean()), "bright_frac": float((g > 250).mean()),
            "unique_colors_sampled": int(len(np.unique(img[::17, ::17].reshape(-1, 3), axis=0)))}
