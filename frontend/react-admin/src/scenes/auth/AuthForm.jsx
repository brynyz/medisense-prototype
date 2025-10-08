import React, { useState } from "react";
import ReCAPTCHA from "react-google-recaptcha";
import {
  Box,
  Button,
  TextField,
  Typography,
  Paper,
  Alert,
  CircularProgress,
  Link,
  Grid,
  useTheme,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { tokens } from "../../theme";

const AuthForm = () => {
  console.log("API URL:", process.env.REACT_APP_API_URL);
  const [captchaValue, setCaptchaValue] = useState(null);

  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const navigate = useNavigate();
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    first_name: "",
    last_name: "",
    password: "",
    password_confirm: "",
  });
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState("");
  const [success, setSuccess] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
    setErrors((prev) => ({
      ...prev,
      [name]: "",
    }));
    setApiError("");
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.username.trim()) {
      newErrors.username = "Username is required";
    } else if (!isLogin && formData.username.length < 3) {
      newErrors.username = "Username must be at least 3 characters";
    }

    if (!isLogin && !formData.email.trim()) {
      newErrors.email = "Email is required";
    } else if (!isLogin && !/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = "Email is invalid";
    }

    if (!isLogin && !formData.first_name.trim()) {
      newErrors.first_name = "First name is required";
    }

    if (!isLogin && !formData.last_name.trim()) {
      newErrors.last_name = "Last name is required";
    }

    if (!formData.password) {
      newErrors.password = "Password is required";
    } else if (!isLogin && formData.password.length < 8) {
      newErrors.password = "Password must be at least 8 characters";
    }

    if (!isLogin && !formData.password_confirm) {
      newErrors.password_confirm = "Password confirmation is required";
    } else if (!isLogin && formData.password !== formData.password_confirm) {
      newErrors.password_confirm = "Passwords do not match";
    }
    if (!captchaValue) {
      newErrors.captcha = "Please complete the reCAPTCHA verification";
      setApiError("Please complete the reCAPTCHA verification");
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleAuth = async (e) => {
    e.preventDefault();
    setApiError("");

    if (!validateForm()) {
      return;
    }

    setLoading(true);

    const endpoint = isLogin
      ? `${process.env.REACT_APP_API_URL}/api/auth/login/`
      : `${process.env.REACT_APP_API_URL}/api/auth/register/`;

    try {
      const payload = isLogin
        ? { username: formData.username, password: formData.password }
        : formData;

      const response = await axios.post(endpoint, payload);

      if (isLogin) {
        localStorage.setItem("token", response.data.access);
        localStorage.setItem("refresh_token", response.data.refresh);
        localStorage.setItem("user", JSON.stringify(response.data.user));
        navigate("/dashboard");
      } else {
        setSuccess(true);
        setTimeout(() => {
          setIsLogin(true); // Switch to login view after successful registration
          setSuccess(false);
          setFormData({
            username: "",
            email: "",
            first_name: "",
            last_name: "",
            password: "",
            password_confirm: "",
          }); // Clear form data
        }, 2000);
      }
    } catch (error) {
      if (error.response?.data) {
        setApiError(error.response.data.detail || "Authentication failed");
      } else {
        setApiError("Network error. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleToggleForm = () => {
    setIsLogin(!isLogin);
    setCaptchaValue(null); 
    setErrors({});
    setApiError("");
    setFormData({
      username: "",
      email: "",
      first_name: "",
      last_name: "",
      password: "",
      password_confirm: "",
    });
  };

  if (success) {
    return (
      <div className="auth-container">
        <Box
          sx={{
            minHeight: "100vh",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            backgroundColor: colors.primary[500],
            padding: 2,
          }}
        >
          <Paper
            elevation={3}
            sx={{
              padding: 4,
              width: "100%",
              maxWidth: 400,
              borderRadius: 2,
              textAlign: "center",
              backgroundColor: colors.primary[400],
              border: `1px solid ${colors.grey[700]}`,
            }}
          >
            <Alert
              severity="success"
              sx={{
                mb: 2,
                backgroundColor: colors.greenAccent[800],
                color: colors.greenAccent[200],
                "& .MuiAlert-icon": {
                  color: colors.greenAccent[200],
                },
              }}
            >
              Registration successful! Redirecting to login...
            </Alert>
            <CircularProgress sx={{ color: colors.greenAccent[500] }} />
          </Paper>
        </Box>
      </div>
    );
  }

  return (
    <div className="auth-container">
      <Box
        sx={{
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: colors.primary[500],
          padding: 2,
        }}
      >
        <Paper
          elevation={3}
          sx={{
            padding: 4,
            width: "100%",
            maxWidth: isLogin ? 400 : 500,
            borderRadius: 2,
            backgroundColor: colors.primary[400],
            border: `1px solid ${colors.grey[700]}`,
          }}
        >
          <Box sx={{ textAlign: "center", mb: 3 }}>
            <Typography
              variant="h4"
              component="h1"
              gutterBottom
              sx={{
                color: colors.grey[100],
                fontWeight: "bold",
              }}
            >
              MediSense
            </Typography>
            <Typography
              variant="h6"
              sx={{
                color: colors.grey[300],
              }}
            >
              {isLogin ? "Sign in to your account" : "Create your account"}
            </Typography>
          </Box>

          {apiError && (
            <Alert
              severity="error"
              sx={{
                mb: 2,
                backgroundColor: colors.redAccent[800],
                color: colors.redAccent[200],
                "& .MuiAlert-icon": {
                  color: colors.redAccent[200],
                },
              }}
            >
              {apiError}
            </Alert>
          )}

          <Box component="form" onSubmit={handleAuth}>
            {!isLogin && (
              <>
                <TextField
                  fullWidth
                  label="First Name"
                  name="first_name"
                  value={formData.first_name}
                  onChange={handleChange}
                  error={!!errors.first_name}
                  helperText={errors.first_name}
                  margin="normal"
                  required
                  variant="filled"
                  sx={{
                    "& .MuiFilledInput-root": {
                      backgroundColor: colors.primary[500],
                      "&:hover": {
                        backgroundColor: colors.primary[400],
                      },
                      "&.Mui-focused": {
                        backgroundColor: colors.primary[400],
                      },
                    },
                    "& .MuiInputLabel-root": {
                      color: colors.grey[200],
                      "&.Mui-focused": {
                        color: colors.blueAccent[500],
                      },
                    },
                    "& .MuiInputBase-input": {
                      color:
                        theme.palette.mode === "dark"
                          ? colors.grey[100]
                          : colors.grey[800],
                      "&::placeholder": {
                        color: "#fcfcfc",
                        opacity: 1,
                      },
                    },
                    "& .MuiFormHelperText-root": {
                      color: colors.redAccent[400],
                    },
                  }}
                />
                <TextField
                  fullWidth
                  label="Last Name"
                  name="last_name"
                  value={formData.last_name}
                  onChange={handleChange}
                  error={!!errors.last_name}
                  helperText={errors.last_name}
                  margin="normal"
                  required
                  variant="filled"
                  sx={{
                    "& .MuiFilledInput-root": {
                      backgroundColor: colors.primary[500],
                      "&:hover": {
                        backgroundColor: colors.primary[400],
                      },
                      "&.Mui-focused": {
                        backgroundColor: colors.primary[400],
                      },
                    },
                    "& .MuiInputLabel-root": {
                      color: colors.grey[200],
                      "&.Mui-focused": {
                        color: colors.blueAccent[500],
                      },
                    },
                    "& .MuiInputBase-input": {
                      color:
                        theme.palette.mode === "dark"
                          ? colors.grey[100]
                          : colors.grey[800],
                      "&::placeholder": {
                        color: "#fcfcfc",
                        opacity: 1,
                      },
                    },
                    "& .MuiFormHelperText-root": {
                      color: colors.redAccent[400],
                    },
                  }}
                />
              </>
            )}

            <TextField
              fullWidth
              label="Username"
              name="username"
              value={formData.username}
              onChange={handleChange}
              error={!!errors.username}
              helperText={errors.username}
              margin="normal"
              required
              autoFocus
              autoComplete="off"
              variant="filled"
              sx={{
                "& .MuiFilledInput-root": {
                  backgroundColor: colors.primary[500],
                  "&:hover": {
                    backgroundColor: colors.primary[400],
                  },
                  "&.Mui-focused": {
                    backgroundColor: colors.primary[400],
                  },
                },
                "& .MuiInputLabel-root": {
                  color: colors.grey[200],
                  "&.Mui-focused": {
                    color: colors.blueAccent[500],
                  },
                },
                "& .MuiInputBase-input": {
                  color:
                    theme.palette.mode === "dark"
                      ? colors.grey[100]
                      : colors.grey[800],
                  "&::placeholder": {
                    color: "#fcfcfc",
                    opacity: 1,
                  },
                },
                "& .MuiFormHelperText-root": {
                  color: colors.redAccent[400],
                },
              }}
            />

            {!isLogin && (
              <TextField
                fullWidth
                label="Email"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                error={!!errors.email}
                helperText={errors.email}
                margin="normal"
                required
                variant="filled"
                sx={{
                  "& .MuiFilledInput-root": {
                    backgroundColor: colors.primary[500],
                    "&:hover": {
                      backgroundColor: colors.primary[400],
                    },
                    "&.Mui-focused": {
                      backgroundColor: colors.primary[400],
                    },
                  },
                  "& .MuiInputLabel-root": {
                    color: colors.grey[200],
                    "&.Mui-focused": {
                      color: colors.blueAccent[500],
                    },
                  },
                  "& .MuiInputBase-input": {
                    color:
                      theme.palette.mode === "dark"
                        ? colors.grey[100]
                        : colors.grey[800],
                    "&::placeholder": {
                      color: "#fcfcfc",
                      opacity: 1,
                    },
                  },
                  "& .MuiFormHelperText-root": {
                    color: colors.redAccent[400],
                  },
                }}
              />
            )}

            <TextField
              fullWidth
              label="Password"
              name="password"
              type="password"
              value={formData.password}
              onChange={handleChange}
              error={!!errors.password}
              helperText={errors.password}
              margin="normal"
              required
              autoComplete="off"
              variant="filled"
              sx={{
                "& .MuiFilledInput-root": {
                  backgroundColor: colors.primary[500],
                  "&:hover": {
                    backgroundColor: colors.primary[400],
                  },
                  "&.Mui-focused": {
                    backgroundColor: colors.primary[400],
                  },
                },
                "& .MuiInputLabel-root": {
                  color: colors.grey[200],
                  "&.Mui-focused": {
                    color: colors.blueAccent[500],
                  },
                },
                "& .MuiInputBase-input": {
                  color:
                    theme.palette.mode === "dark"
                      ? colors.grey[100]
                      : colors.grey[800],
                  "&::placeholder": {
                    color: "#fcfcfc",
                    opacity: 1,
                  },
                },
                "& .MuiFormHelperText-root": {
                  color: colors.redAccent[400],
                },
              }}
            />

            {!isLogin && (
              <TextField
                fullWidth
                label="Confirm Password"
                name="password_confirm"
                type="password"
                value={formData.password_confirm}
                onChange={handleChange}
                error={!!errors.password_confirm}
                helperText={errors.password_confirm}
                margin="normal"
                required
                variant="filled"
                sx={{
                  "& .MuiFilledInput-root": {
                    backgroundColor: colors.primary[500],
                    "&:hover": {
                      backgroundColor: colors.primary[400],
                    },
                    "&.Mui-focused": {
                      backgroundColor: colors.primary[400],
                    },
                  },
                  "& .MuiInputLabel-root": {
                    color: colors.grey[200],
                    "&.Mui-focused": {
                      color: colors.blueAccent[500],
                    },
                  },
                  "& .MuiInputBase-input": {
                    color:
                      theme.palette.mode === "dark"
                        ? colors.grey[100]
                        : colors.grey[800],
                    "&::placeholder": {
                      color: "#fcfcfc",
                      opacity: 1,
                    },
                  },
                  "& .MuiFormHelperText-root": {
                    color: colors.redAccent[400],
                  },
                }}
              />
            )}
              <Box
                sx={{ mt: 2, mb: 2, display: "flex", justifyContent: "center" }}
              >
                <ReCAPTCHA
                  sitekey="6LcH1-ErAAAAAJgoNa8dqcYX1hlmmmgdJ6j5MC7E"
                  onChange={setCaptchaValue}
                  theme={theme.palette.mode === "dark" ? "dark" : "light"}
                />
                {errors.captcha && (
                  <Typography color="error" variant="caption" sx={{ mt: 1 }}>
                    {errors.captcha}
                  </Typography>
                )}
              </Box>
            <Button
              type="submit"
              fullWidth
              variant="contained"
              disabled={loading}
              sx={{
                mt: 3,
                mb: 2,
                py: 1.5,
                backgroundColor: colors.blueAccent[600],
                color: colors.grey[100],
                "&:hover": {
                  backgroundColor: colors.blueAccent[700],
                },
                "&:disabled": {
                  backgroundColor: colors.grey[600],
                  color: colors.grey[400],
                },
              }}
            >
              {loading ? (
                <CircularProgress size={24} sx={{ color: colors.grey[100] }} />
              ) : isLogin ? (
                "Sign In"
              ) : (
                "Sign Up"
              )}
            </Button>

            <Box sx={{ textAlign: "center" }}>
              <Typography variant="body2" sx={{ color: colors.grey[300] }}>
                {isLogin
                  ? "Don't have an account?"
                  : "Already have an account?"}{" "}
                <Link
                  component="button"
                  type="button"
                  onClick={handleToggleForm}
                  sx={{
                    textDecoration: "none",
                    color: colors.blueAccent[500],
                    "&:hover": {
                      color: colors.blueAccent[400],
                      textDecoration: "underline",
                    },
                  }}
                >
                  {isLogin ? "Sign up" : "Sign in"}
                </Link>
              </Typography>
            </Box>
          </Box>
        </Paper>
      </Box>
    </div>
  );
};

export default AuthForm;
