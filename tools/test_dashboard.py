"""Test the web dashboard — starts server, verifies all pages render without errors."""
import subprocess, time, sys, urllib.request, urllib.error

def main():
    # Start the server in background
    proc = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'web.app:app', '--port', '8102', '--log-level', 'error'],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    time.sleep(4)

    # Check if process is running
    if proc.poll() is not None:
        print('SERVER FAILED TO START')
        stderr = proc.stderr.read().decode('utf-8', errors='replace')
        print(stderr[:2000])
        return 1

    print('Server started successfully on port 8102')

    # Test endpoints
    tests = [
        ('/', 'Dashboard HTML page'),
        ('/future-predictions', 'Future predictions HTML page'),
        ('/api/health', 'Health endpoint'),
    ]

    all_ok = True
    for path, desc in tests:
        url = f'http://127.0.0.1:8102{path}'
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode('utf-8', errors='replace')
                status = resp.status
            if status == 200:
                print(f'  [OK] {desc}: HTTP {status}, {len(body)} bytes')
            else:
                print(f'  [FAIL] {desc}: HTTP {status}')
                all_ok = False
        except urllib.error.HTTPError as e:
            print(f'  [FAIL] {desc}: HTTP {e.code}')
            # Try to read error body
            try:
                err_body = e.read().decode('utf-8', errors='replace')
                if 'detail' in err_body.lower() or 'error' in err_body.lower():
                    print(f'  Error detail: {err_body[:500]}')
            except Exception:
                pass
            all_ok = False
        except Exception as e:
            print(f'  [FAIL] {desc}: Exception: {e}')
            all_ok = False

    # Check static files
    static_tests = [
        '/static/css/app.css',
        '/static/css/future-predictions.css',
        '/static/js/app.js',
    ]
    for path in static_tests:
        url = f'http://127.0.0.1:8102{path}'
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode('utf-8', errors='replace')
                status = resp.status
            if status == 200:
                print(f'  [OK] Static {path}: HTTP {status}, {len(body)} bytes')
            else:
                print(f'  [FAIL] Static {path}: HTTP {status}')
                all_ok = False
        except urllib.error.HTTPError as e:
            print(f'  [FAIL] Static {path}: HTTP {e.code}')
            all_ok = False
        except Exception as e:
            print(f'  [FAIL] Static {path}: Exception: {e}')
            all_ok = False

    # Check no Jinja2 template errors in rendered HTML
    try:
        req = urllib.request.Request('http://127.0.0.1:8102/')
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode('utf-8', errors='replace')

        jinja_errors = ['UndefinedError', 'TemplateNotFound', 'TemplateSyntaxError', 'jinja2.exceptions']
        for err in jinja_errors:
            if err in body:
                print(f'  [FAIL] Template error found: {err}')
                all_ok = False

        if 'Internal Server Error' in body:
            print('  [FAIL] Internal Server Error in dashboard HTML')
            all_ok = False
    except Exception as e:
        print(f'  [FAIL] Could not check template errors: {e}')
        all_ok = False

    print()
    if all_ok:
        print('ALL CHECKS PASSED - All pages and static assets render correctly')
    else:
        print('SOME CHECKS FAILED')

    # Cleanup
    proc.terminate()
    try:
        proc.wait(3)
    except Exception:
        proc.kill()
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())
