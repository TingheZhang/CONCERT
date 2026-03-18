diff --git a/src/CONCERT-long/concert_batch_gut.py b/src/CONCERT-long/concert_batch_gut.py
index 8ca7e7379239d6c9acd0038c8fdbc9755d2a9e9e..ae26b080bdc59b21ba9894bb2fa0f6c5f156e298 100644
--- a/src/CONCERT-long/concert_batch_gut.py
+++ b/src/CONCERT-long/concert_batch_gut.py
@@ -72,69 +72,77 @@ class EarlyStopping:
 # ---------------------------------------------------------------------
 class CONCERT(nn.Module):
     """
     CONCERT — GP-VAE for the gut dataset with **day-batched** SVGP kernel.
     """
 
     def __init__(
         self,
         *,
         input_dim: int,
         GP_dim: int,
         Normal_dim: int,
         cell_atts: np.ndarray,
         num_genes: int,
         n_batch: int,
         encoder_layers: Iterable[int],
         decoder_layers: Iterable[int],
         noise: float,
         encoder_dropout: float,
         decoder_dropout: float,
         shared_dispersion: bool,
         fixed_inducing_points: bool,
         initial_inducing_points: np.ndarray,  # shape (M*n_batch, 2 + n_batch) for batched kernel
         fixed_gp_params: bool,
         kernel_scale: float | np.ndarray,     # float or (n_batch, spatial_dims) if batched
-        multi_kernel_mode: bool,
+        multi_kernel_mode: bool | None = None,
+        allow_batch_kernel_scale: bool = False,
         N_train: int,
         KL_loss: float,
         dynamicVAE: bool,
         init_beta: float,
         min_beta: float,
         max_beta: float,
         dtype: torch.dtype,
         device: str,
     ) -> None:
         super().__init__()
         torch.set_default_dtype(dtype)
 
+        effective_multi_kernel_mode = bool(allow_batch_kernel_scale) if multi_kernel_mode is None else bool(multi_kernel_mode)
+        effective_kernel_scale = (
+            [kernel_scale] * n_batch
+            if effective_multi_kernel_mode and np.isscalar(kernel_scale)
+            else kernel_scale
+        )
+
         self.svgp = SVGP(
             fixed_inducing_points=fixed_inducing_points,
             initial_inducing_points=initial_inducing_points,
             fixed_gp_params=fixed_gp_params,
-            kernel_scale=kernel_scale,
-            multi_kernel_mode=multi_kernel_mode,
+            kernel_scale=effective_kernel_scale,
+            multi_kernel_mode=effective_multi_kernel_mode,
             jitter=1e-8,
             N_train=N_train,
             dtype=dtype,
             device=device,
         )
 
         # Hyperparams
         self.input_dim = int(input_dim)
         self.PID = PIDControl(Kp=0.01, Ki=-0.005, init_beta=init_beta, min_beta=min_beta, max_beta=max_beta)
         self.KL_loss_target = float(KL_loss)
         self.dynamicVAE = bool(dynamicVAE)
         self.beta = float(init_beta)
         self.dtype = dtype
         self.GP_dim = int(GP_dim)
         self.Normal_dim = int(Normal_dim)
         self.noise = float(noise)
         self.device = device
         self.num_genes = int(num_genes)
         self.cell_atts = cell_atts
         self.n_batch = int(n_batch)
         self.shared_dispersion = bool(shared_dispersion)
 
         # LORD encoder: attributes = ["perturbation"(categorical), "day"(ordinal)]
         self.lord_encoder = LordEncoder(
             embedding_dim=[64, 64, 64],
diff --git a/tests/test_gut_notebook_constructor_compat.py b/tests/test_gut_notebook_constructor_compat.py
new file mode 100644
index 0000000000000000000000000000000000000000..42662a5252a5e65e7d86959df415c32350dac9d2
--- /dev/null
+++ b/tests/test_gut_notebook_constructor_compat.py
@@ -0,0 +1,88 @@
+import ast
+import json
+import unittest
+from pathlib import Path
+
+
+REPO_ROOT = Path(__file__).resolve().parents[1]
+CONSTRUCTOR_FILE = REPO_ROOT / 'src' / 'CONCERT-long' / 'concert_batch_gut.py'
+NOTEBOOK_FILE = REPO_ROOT / 'notebooks' / 'run_concert_mouse_colon.ipynb'
+RUNNER_FILE = REPO_ROOT / 'src' / 'CONCERT-long' / 'run_concert_gut.py'
+
+
+class ConstructorKwargVisitor(ast.NodeVisitor):
+    def __init__(self) -> None:
+        self.kwonlyargs: list[str] = []
+
+    def visit_ClassDef(self, node: ast.ClassDef) -> None:
+        if node.name != 'CONCERT':
+            return
+        for item in node.body:
+            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
+                self.kwonlyargs = [arg.arg for arg in item.args.kwonlyargs]
+                return
+
+
+class CallKwargVisitor(ast.NodeVisitor):
+    def __init__(self) -> None:
+        self.keywords: list[str] = []
+
+    def visit_Assign(self, node: ast.Assign) -> None:
+        if not isinstance(node.value, ast.Call):
+            return
+        if not isinstance(node.value.func, ast.Name) or node.value.func.id != 'CONCERT':
+            return
+        self.keywords = [kw.arg for kw in node.value.keywords if kw.arg is not None]
+
+
+class GutNotebookConstructorCompatTest(unittest.TestCase):
+    def _constructor_kwonlyargs(self) -> list[str]:
+        constructor_tree = ast.parse(CONSTRUCTOR_FILE.read_text())
+        constructor_visitor = ConstructorKwargVisitor()
+        constructor_visitor.visit(constructor_tree)
+        self.assertTrue(constructor_visitor.kwonlyargs, 'Failed to locate CONCERT.__init__ kw-only args.')
+        return constructor_visitor.kwonlyargs
+
+    def test_mouse_colon_notebook_constructor_kwargs_are_supported(self) -> None:
+        constructor_kwonlyargs = self._constructor_kwonlyargs()
+
+        notebook = json.loads(NOTEBOOK_FILE.read_text())
+        notebook_sources = []
+        for cell in notebook['cells']:
+            if cell.get('cell_type') == 'code':
+                notebook_sources.append(''.join(cell.get('source', [])))
+
+        notebook_tree = ast.parse('\n\n'.join(notebook_sources))
+        notebook_visitor = CallKwargVisitor()
+        notebook_visitor.visit(notebook_tree)
+        self.assertTrue(notebook_visitor.keywords, 'Failed to locate CONCERT(...) notebook call.')
+
+        unsupported_kwargs = sorted(set(notebook_visitor.keywords) - set(constructor_kwonlyargs))
+        self.assertEqual(
+            unsupported_kwargs,
+            [],
+            f'Notebook passes unsupported CONCERT kwargs: {unsupported_kwargs}',
+        )
+
+        self.assertIn('allow_batch_kernel_scale', constructor_kwonlyargs)
+        self.assertIn('multi_kernel_mode', constructor_kwonlyargs)
+
+    def test_run_concert_gut_constructor_kwargs_are_supported(self) -> None:
+        constructor_kwonlyargs = self._constructor_kwonlyargs()
+        runner_tree = ast.parse(RUNNER_FILE.read_text())
+        runner_visitor = CallKwargVisitor()
+        runner_visitor.visit(runner_tree)
+        self.assertTrue(runner_visitor.keywords, 'Failed to locate CONCERT(...) call in run_concert_gut.py.')
+
+        unsupported_kwargs = sorted(set(runner_visitor.keywords) - set(constructor_kwonlyargs))
+        self.assertEqual(
+            unsupported_kwargs,
+            [],
+            f'run_concert_gut.py passes unsupported CONCERT kwargs: {unsupported_kwargs}',
+        )
+
+        self.assertIn('multi_kernel_mode', runner_visitor.keywords)
+
+
+if __name__ == '__main__':
+    unittest.main()
