use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

use vackrooms::entities::models::Position;
use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
use vackrooms::frameworks_drivers::std_telemetry::StdTelemetry;
use vackrooms::interface_adapters::web_renderer::WebRendererAdapter;
use vackrooms::use_cases::generate_chunk::GenerateChunkArchitectureUseCase;

/// Presentation Layer & Driver:
/// A minimalist, zero-dependency HTTP server that connects our
/// Clean Architecture engine to the Browser Voxel Renderer.
fn main() {
    let port = 8080;
    let listener = TcpListener::bind(format!("127.0.0.1:{}", port)).unwrap();
    println!("--- Backrooms Engine Running ---");
    println!("Server listening on http://localhost:{}", port);
    println!("Open the above URL in your browser to view the Voxel Rendering.");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                thread::spawn(move || {
                    handle_client(stream);
                });
            }
            Err(e) => {
                eprintln!("Failed to accept connection: {}", e);
            }
        }
    }
}

fn handle_client(mut stream: TcpStream) {
    let mut buffer = [0; 1024];
    if let Ok(bytes_read) = stream.read(&mut buffer) {
        if bytes_read == 0 {
            return;
        }

        let request = String::from_utf8_lossy(&buffer[..bytes_read]);
        let request_line = request.lines().next().unwrap_or("");

        if request_line.starts_with("GET /maze") {
            // Serve the Procedural Voxel Data with support for coordinate query parameters
            let parts: Vec<&str> = request_line.split_whitespace().collect();
            let url = parts.get(1).unwrap_or(&"/maze");

            let mut chunk_x = 0.0;
            let mut chunk_z = 0.0;
            let mut spec = "high";

            if let Some(query_idx) = url.find('?') {
                let query = &url[query_idx + 1..];
                for param in query.split('&') {
                    let key_val: Vec<&str> = param.split('=').collect();
                    if key_val.len() == 2 {
                        match key_val[0] {
                            "x" => chunk_x = key_val[1].parse().unwrap_or(0.0),
                            "z" => chunk_z = key_val[1].parse().unwrap_or(0.0),
                            "spec" => spec = key_val[1],
                            _ => {}
                        }
                    }
                }
            }

            let config = if spec == "low" {
                vackrooms::use_cases::generate_chunk::GeneratorConfig::low_spec()
            } else {
                vackrooms::use_cases::generate_chunk::GeneratorConfig::high_spec()
            };

            let noise_provider = SimpleNoiseProvider::new();
            let telemetry = StdTelemetry;
            let generator =
                GenerateChunkArchitectureUseCase::with_telemetry(&noise_provider, &telemetry);

            let chunk = generator.execute(Position::new(chunk_x, chunk_z), 42, config);

            // Adapt to Web Format (greedy meshed faces)
            let json = WebRendererAdapter::to_json(&chunk, config.chunk_size);

            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                json.len(),
                json
            );
            let _ = stream.write_all(response.as_bytes());
        } else if request_line.starts_with("GET /octree") {
            // Serve the SVO Packed Data
            let parts: Vec<&str> = request_line.split_whitespace().collect();
            let url = parts.get(1).unwrap_or(&"/octree");

            let mut chunk_x = 0.0;
            let mut chunk_z = 0.0;
            let mut spec = "high";

            if let Some(query_idx) = url.find('?') {
                let query = &url[query_idx + 1..];
                for param in query.split('&') {
                    let key_val: Vec<&str> = param.split('=').collect();
                    if key_val.len() == 2 {
                        match key_val[0] {
                            "x" => chunk_x = key_val[1].parse().unwrap_or(0.0),
                            "z" => chunk_z = key_val[1].parse().unwrap_or(0.0),
                            "spec" => spec = key_val[1],
                            _ => {}
                        }
                    }
                }
            }

            let config = if spec == "low" {
                vackrooms::use_cases::generate_chunk::GeneratorConfig::low_spec()
            } else {
                vackrooms::use_cases::generate_chunk::GeneratorConfig::high_spec()
            };

            let noise_provider = SimpleNoiseProvider::new();
            let telemetry = StdTelemetry;
            let generator =
                GenerateChunkArchitectureUseCase::with_telemetry(&noise_provider, &telemetry);

            // Generate chunk at position with GeneratorConfig
            let chunk = generator.execute(Position::new(chunk_x, chunk_z), 42, config);

            // Adapt to Web Format using SVO representation
            let depth = config.svo_depth();
            let world_size = config.svo_world_size();
            let binary = WebRendererAdapter::to_octree_binary(
                &chunk,
                depth,
                world_size,
                config.voxel_scale,
                config.chunk_size,
            );

            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\n\r\n",
                binary.len()
            );
            let mut response = Vec::with_capacity(header.len() + binary.len());
            response.extend_from_slice(header.as_bytes());
            response.extend_from_slice(&binary);
            let _ = stream.write_all(&response);
        } else if request_line.starts_with("GET ") {
            // Static file service for the wasm front end (and the legacy page).
            let parts: Vec<&str> = request_line.split_whitespace().collect();
            let url = parts.get(1).unwrap_or(&"/");
            let path = url.split('?').next().unwrap_or("/");
            serve_static(&mut stream, path);
        } else {
            let error = "HTTP/1.1 404 Not Found\r\n\r\n";
            let _ = stream.write_all(error.as_bytes());
        }
    }
}

/// Maps a URL path to a file under `static/` and serves it with the correct
/// MIME type. `application/wasm` matters: browsers refuse to instantiate
/// wasm streams served under the wrong content type.
fn serve_static(stream: &mut TcpStream, url_path: &str) {
    let relative = match url_path {
        "/" | "/index.html" => "index.html",
        "/legacy" => "legacy_threejs.html",
        other => other.trim_start_matches('/'),
    };

    // Path traversal guard: reject anything trying to escape `static/`.
    if relative.split('/').any(|seg| seg == "..") || relative.contains('\\') {
        let _ = stream.write_all(b"HTTP/1.1 403 Forbidden\r\n\r\n");
        return;
    }

    let file_path = format!("static/{}", relative);
    match fs::read(&file_path) {
        Ok(body) => {
            let mime = match file_path.rsplit('.').next() {
                Some("html") => "text/html; charset=utf-8",
                Some("js") | Some("mjs") => "text/javascript",
                Some("wasm") => "application/wasm",
                Some("css") => "text/css",
                Some("json") => "application/json",
                Some("png") => "image/png",
                _ => "application/octet-stream",
            };
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\n\r\n",
                mime,
                body.len()
            );
            let mut response = Vec::with_capacity(header.len() + body.len());
            response.extend_from_slice(header.as_bytes());
            response.extend_from_slice(&body);
            let _ = stream.write_all(&response);
        }
        Err(_) => {
            let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\n\r\nFile not found.");
        }
    }
}
