#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commdlg.h>
#else
#include <pty.h>
#include <gtk/gtk.h>
#endif

#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

namespace {

constexpr const char* kDefaultVocabPath = "./rwkv_vocab_v20230424.txt";
constexpr int kDefaultPort = 8000;
constexpr std::size_t kMaxTerminalBytes = 16u << 20;
constexpr const char* kBackendExecutable = "./rwkv_lighting_cuda";

std::optional<std::string> open_native_file_dialog(
    const std::string& title,
    const std::string& current_path) {
#ifdef _WIN32
  char buffer[4096] = {};
  if (!current_path.empty()) {
    std::snprintf(buffer, sizeof(buffer), "%s", current_path.c_str());
  }

  OPENFILENAMEA ofn{};
  ofn.lStructSize = sizeof(ofn);
  ofn.hwndOwner = nullptr;
  ofn.lpstrFile = buffer;
  ofn.nMaxFile = sizeof(buffer);
  ofn.lpstrTitle = title.c_str();
  ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
  if (GetOpenFileNameA(&ofn) == TRUE) {
    return std::string(buffer);
  }
  return std::nullopt;
#else
  static bool gtk_initialized = gtk_init_check(nullptr, nullptr);
  if (!gtk_initialized) {
    return std::nullopt;
  }

  GtkFileChooserNative* chooser = gtk_file_chooser_native_new(
      title.c_str(),
      nullptr,
      GTK_FILE_CHOOSER_ACTION_OPEN,
      "_Open",
      "_Cancel");
  GtkFileChooser* file_chooser = GTK_FILE_CHOOSER(chooser);

  const std::filesystem::path path_hint =
      current_path.empty() ? std::filesystem::current_path() : std::filesystem::path(current_path);
  if (!current_path.empty() && std::filesystem::exists(path_hint) && std::filesystem::is_regular_file(path_hint)) {
    gtk_file_chooser_set_filename(file_chooser, path_hint.c_str());
  } else {
    const std::filesystem::path folder_hint =
        std::filesystem::is_directory(path_hint) ? path_hint : path_hint.parent_path();
    if (!folder_hint.empty() && std::filesystem::exists(folder_hint)) {
      gtk_file_chooser_set_current_folder(file_chooser, folder_hint.c_str());
    }
  }

  std::optional<std::string> selected_path;
  const int response = gtk_native_dialog_run(GTK_NATIVE_DIALOG(chooser));
  if (response == GTK_RESPONSE_ACCEPT) {
    char* filename = gtk_file_chooser_get_filename(file_chooser);
    if (filename) {
      selected_path = filename;
      g_free(filename);
    }
  }

  g_object_unref(chooser);
  while (gtk_events_pending()) {
    gtk_main_iteration();
  }
  return selected_path;
#endif
}

struct ManagedProcess {
  pid_t pid = -1;
  int stdout_fd = -1;
  std::string text;
  std::vector<int> line_offsets{0};
  int exit_code = 0;
  bool running = false;
  bool follow_output = true;
  bool scroll_to_bottom = false;

  ~ManagedProcess() {
    stop(true);
  }

  bool start(
      const std::string& model_path,
      const std::string& vocab_path,
      int port,
      const std::string& password) {
    if (running) {
      append_line("process is already running");
      return false;
    }

    std::vector<std::string> args;
    args.emplace_back(kBackendExecutable);
    args.emplace_back("--model-path");
    args.emplace_back(model_path);
    args.emplace_back("--vocab-path");
    args.emplace_back(vocab_path.empty() ? kDefaultVocabPath : vocab_path);
    args.emplace_back("--port");
    args.emplace_back(std::to_string(port));
    if (!password.empty()) {
      args.emplace_back("--password");
      args.emplace_back(password);
    }

    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& arg : args) {
      argv.push_back(arg.data());
    }
    argv.push_back(nullptr);

#ifdef _WIN32
    int pipe_fds[2] = {-1, -1};
    if (pipe(pipe_fds) != 0) {
      append_line(std::string("pipe failed: ") + std::strerror(errno));
      return false;
    }

    const pid_t child = fork();
    if (child < 0) {
      append_line(std::string("fork failed: ") + std::strerror(errno));
      close(pipe_fds[0]);
      close(pipe_fds[1]);
      return false;
    }

    if (child == 0) {
      dup2(pipe_fds[1], STDOUT_FILENO);
      dup2(pipe_fds[1], STDERR_FILENO);
      close(pipe_fds[0]);
      close(pipe_fds[1]);
      execv(argv[0], argv.data());

      std::fprintf(stderr, "execv failed: %s\n", std::strerror(errno));
      _exit(127);
    }

    close(pipe_fds[1]);
    const int flags = fcntl(pipe_fds[0], F_GETFL, 0);
    if (flags >= 0) {
      fcntl(pipe_fds[0], F_SETFL, flags | O_NONBLOCK);
    }

    pid = child;
    stdout_fd = pipe_fds[0];
#else
    int master_fd = -1;
    const pid_t child = forkpty(&master_fd, nullptr, nullptr, nullptr);
    if (child < 0) {
      append_line(std::string("forkpty failed: ") + std::strerror(errno));
      return false;
    }
    if (child == 0) {
      execv(argv[0], argv.data());
      std::fprintf(stderr, "execv failed: %s\n", std::strerror(errno));
      _exit(127);
    }

    const int flags = fcntl(master_fd, F_GETFL, 0);
    if (flags >= 0) {
      fcntl(master_fd, F_SETFL, flags | O_NONBLOCK);
    }

    pid = child;
    stdout_fd = master_fd;
#endif
    running = true;
    exit_code = 0;
    clear_output();
    append_line(std::string("started: ") + kBackendExecutable);
    return true;
  }

  void read_available_output() {
    if (stdout_fd < 0) {
      return;
    }
    std::array<char, 4096> chunk{};
    while (true) {
      const ssize_t n = read(stdout_fd, chunk.data(), chunk.size());
      if (n > 0) {
        append_text(chunk.data(), static_cast<std::size_t>(n));
        continue;
      }
      if (n == 0) {
        close(stdout_fd);
        stdout_fd = -1;
        break;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        break;
      }
#ifndef _WIN32
      if (errno == EIO) {
        close(stdout_fd);
        stdout_fd = -1;
        break;
      }
#endif
      append_line(std::string("read failed: ") + std::strerror(errno));
      close(stdout_fd);
      stdout_fd = -1;
      break;
    }
  }

  void drain_output_after_exit() {
    if (stdout_fd < 0) {
      return;
    }
    for (int attempt = 0; attempt < 100 && stdout_fd >= 0; ++attempt) {
      const std::size_t before = text.size();
      read_available_output();
      if (stdout_fd < 0) {
        break;
      }
      if (text.size() == before) {
        usleep(10 * 1000);
      }
    }
  }

  void pump_output() {
    read_available_output();

    if (!running || pid <= 0) {
      return;
    }

    int status = 0;
    const pid_t result = waitpid(pid, &status, WNOHANG);
    if (result == 0) {
      return;
    }
    if (result < 0) {
      append_line(std::string("waitpid failed: ") + std::strerror(errno));
      running = false;
      pid = -1;
      return;
    }

    running = false;
    pid = -1;
    drain_output_after_exit();
    if (WIFEXITED(status)) {
      exit_code = WEXITSTATUS(status);
      append_line("process exited with code " + std::to_string(exit_code));
    } else if (WIFSIGNALED(status)) {
      exit_code = 128 + WTERMSIG(status);
      append_line("process terminated by signal " + std::to_string(WTERMSIG(status)));
    }
  }

  void stop(bool force_kill) {
    if (!running || pid <= 0) {
      if (stdout_fd >= 0) {
        close(stdout_fd);
        stdout_fd = -1;
      }
      return;
    }
    kill(pid, force_kill ? SIGKILL : SIGTERM);
    int status = 0;
    waitpid(pid, &status, 0);
    running = false;
    pid = -1;
    drain_output_after_exit();
    if (stdout_fd >= 0) {
      close(stdout_fd);
      stdout_fd = -1;
    }
    append_line(force_kill ? "process killed" : "stop signal sent");
  }

  void clear_output() {
    text.clear();
    line_offsets.clear();
    line_offsets.push_back(0);
    scroll_to_bottom = true;
  }

  void append_line(const std::string& line) {
    append_text(line.data(), line.size());
    append_text("\n", 1);
  }

  void append_text(const char* data, std::size_t size) {
    if (size == 0) {
      return;
    }
    const int old_size = static_cast<int>(text.size());
    text.append(data, size);
    for (int i = old_size; i < static_cast<int>(text.size()); ++i) {
      if (text[static_cast<std::size_t>(i)] == '\n') {
        line_offsets.push_back(i + 1);
      }
    }
    trim_output();
    if (follow_output) {
      scroll_to_bottom = true;
    }
  }

  void trim_output() {
    if (text.size() <= kMaxTerminalBytes) {
      return;
    }
    std::size_t trim_to = text.size() - kMaxTerminalBytes;
    const std::size_t newline = text.find('\n', trim_to);
    if (newline != std::string::npos) {
      trim_to = newline + 1;
    }
    text.erase(0, trim_to);
    line_offsets.clear();
    line_offsets.push_back(0);
    for (int i = 0; i < static_cast<int>(text.size()); ++i) {
      if (text[static_cast<std::size_t>(i)] == '\n') {
        line_offsets.push_back(i + 1);
      }
    }
  }

  void draw_terminal() {
    ImGui::Checkbox("Follow Output", &follow_output);
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
      clear_output();
    }
    ImGui::Separator();

    ImGui::BeginChild("terminal_output", ImVec2(0.0f, 0.0f), true, ImGuiWindowFlags_HorizontalScrollbar);
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(line_offsets.size()));
    while (clipper.Step()) {
      for (int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd; ++line_no) {
        const int line_start = line_offsets[static_cast<std::size_t>(line_no)];
        const int line_end =
            (line_no + 1 < static_cast<int>(line_offsets.size()))
                ? (line_offsets[static_cast<std::size_t>(line_no + 1)] - 1)
                : static_cast<int>(text.size());
        if (line_start <= line_end && line_start <= static_cast<int>(text.size())) {
          ImGui::TextUnformatted(text.data() + line_start, text.data() + line_end);
        }
      }
    }
    if (scroll_to_bottom) {
      ImGui::SetScrollHereY(1.0f);
      scroll_to_bottom = false;
    }
    ImGui::EndChild();
  }
};

void draw_labeled_path_row(
    const char* label,
    const char* input_id,
    const char* browse_id,
    std::string& value,
    const char* dialog_title) {
  char buffer[1024];
  std::snprintf(buffer, sizeof(buffer), "%s", value.c_str());

  ImGui::TableNextRow();
  ImGui::TableSetColumnIndex(0);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(label);

  ImGui::TableSetColumnIndex(1);
  ImGui::PushItemWidth(-FLT_MIN);
  if (ImGui::InputText(input_id, buffer, sizeof(buffer))) {
    value = buffer;
  }
  ImGui::PopItemWidth();

  ImGui::TableSetColumnIndex(2);
  if (ImGui::Button(browse_id)) {
    if (auto chosen = open_native_file_dialog(dialog_title, value)) {
      value = *chosen;
    }
  }
}

}  // namespace

int main() {
  if (!glfwInit()) {
    std::fprintf(stderr, "glfwInit failed\n");
    return 1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window = glfwCreateWindow(1040, 760, "RWKV Lighting CUDA Launcher", nullptr, nullptr);
  if (!window) {
    std::fprintf(stderr, "glfwCreateWindow failed\n");
    glfwTerminate();
    return 1;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigWindowsMoveFromTitleBarOnly = true;
  ImGui::StyleColorsDark();

  ImGuiStyle& style = ImGui::GetStyle();
  style.FramePadding = ImVec2(8.0f, 6.0f);
  style.ItemSpacing = ImVec2(10.0f, 8.0f);
  style.WindowPadding = ImVec2(12.0f, 12.0f);

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  std::string model_path;
  std::string vocab_path = kDefaultVocabPath;
  int port = kDefaultPort;
  std::string password;
  ManagedProcess process;

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    process.pump_output();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(io.DisplaySize, ImGuiCond_Always);
    ImGui::Begin(
        "RWKV Lighting CUDA Launcher",
        nullptr,
        ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse);

    if (ImGui::BeginTable("path_table", 3, ImGuiTableFlags_SizingStretchProp)) {
      ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 100.0f);
      ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 92.0f);
      draw_labeled_path_row("Model Path", "##model_path", "Browse##model", model_path, "Select Model Path");
      draw_labeled_path_row("Vocab Path", "##vocab_path", "Browse##vocab", vocab_path, "Select Vocab Path");
      ImGui::EndTable();
    }

    char password_buffer[256];
    std::snprintf(password_buffer, sizeof(password_buffer), "%s", password.c_str());

    if (ImGui::BeginTable("config_table", 4, ImGuiTableFlags_SizingStretchProp)) {
      ImGui::TableSetupColumn("PortLabel", ImGuiTableColumnFlags_WidthFixed, 60.0f);
      ImGui::TableSetupColumn("PortValue", ImGuiTableColumnFlags_WidthFixed, 120.0f);
      ImGui::TableSetupColumn("PasswordLabel", ImGuiTableColumnFlags_WidthFixed, 80.0f);
      ImGui::TableSetupColumn("PasswordValue", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableNextRow();

      ImGui::TableSetColumnIndex(0);
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted("Port");

      ImGui::TableSetColumnIndex(1);
      ImGui::PushItemWidth(-FLT_MIN);
      ImGui::InputInt("##port", &port);
      ImGui::PopItemWidth();
      port = std::clamp(port, 1, 65535);

      ImGui::TableSetColumnIndex(2);
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted("Password");

      ImGui::TableSetColumnIndex(3);
      ImGui::PushItemWidth(-FLT_MIN);
      if (ImGui::InputText("##password", password_buffer, sizeof(password_buffer), ImGuiInputTextFlags_Password)) {
        password = password_buffer;
      }
      ImGui::PopItemWidth();
      ImGui::EndTable();
    }

    ImGui::Separator();

    const float footer_height = ImGui::GetFrameHeightWithSpacing() * 2.0f;
    ImGui::BeginChild("terminal_panel", ImVec2(0.0f, -footer_height), false);
    process.draw_terminal();
    ImGui::EndChild();

    const bool can_start = !process.running && !model_path.empty();
    if (!can_start) {
      ImGui::BeginDisabled();
    }
    if (ImGui::Button("Start", ImVec2(120.0f, 0.0f))) {
      process.start(model_path, vocab_path, port, password);
    }
    if (!can_start) {
      ImGui::EndDisabled();
    }

    ImGui::SameLine();
    if (!process.running) {
      ImGui::BeginDisabled();
    }
    if (ImGui::Button("Stop", ImVec2(120.0f, 0.0f))) {
      process.stop(false);
    }
    if (!process.running) {
      ImGui::EndDisabled();
    }

    ImGui::SameLine();
    ImGui::TextUnformatted(process.running ? "Running" : "Stopped");

    ImGui::End();

    ImGui::Render();
    int display_w = 0;
    int display_h = 0;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.10f, 0.12f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  process.stop(false);
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
