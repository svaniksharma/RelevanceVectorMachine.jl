library("JuliaConnectoR")
library("kernlab")

# Initialize Julia and RelevanceVectorMachine

print(paste0("Your current working directory is: ",
             getwd(), "; Assuming this is path to rvm"))
path_to_rvm <- getwd()
juliaEval(paste0("import Pkg; Pkg.activate(\"", path_to_rvm, "\")"))
juliaEval(paste0("include(\"", getwd(), "/test/julia_r_tests.jl\")"))

# Test functions -- We want our RVM to match the predictions of
# kernlab (approximately)

x <- seq(-1, 1, length.out = 100)
df <- data.frame(
  x = x,
  y1 = -23.4 * x,
  y2 = 1 + 0.2786 * x + 9.874 * x^2,
  y3 = sin(x / 5) + 1 / 3 * x
)

test_synthetic_func <- function(y_col, y_col_name) {
  # Fit the data using kernlab
  rvm_mod <- rvm(x, y = y_col)
  r_result <- as.numeric(predict(rvm_mod, x))
  # Call corresponding fit method for RVM and fit the corresponding data
  julia_result <- juliaLet("fit_synthetic_func(col)", col = y_col_name)
  julia_result <- as.numeric(julia_result)
  # Compare (should be kind-of similar;
  # kernlab uses an RBF kernel, so results will vary)
  all.equal(r_result, julia_result, tolerance = 1e-1)
}

test_synthetic_func(df$y1, "y1")
test_synthetic_func(df$y2, "y2")
test_synthetic_func(df$y3, "y3")