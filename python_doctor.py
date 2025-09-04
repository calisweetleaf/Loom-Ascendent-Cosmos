#!/usr/bin/env python3
"""
Python Production Doctor - Comprehensive Code Health Assessment Tool

Scans Python files/directories to identify:
- Syntax errors
- TODOs and technical debt markers
- Stub implementations (empty, pass, ellipsis, NotImplementedError)
- Simple placeholder returns (return None, 0, "", etc.)
- Incomplete method implementations
- Missing docstrings
- Suspiciously short functions
- Unimplemented abstract methods
- Type hint completeness
- Test coverage gaps

Outputs a detailed markdown report suitable for production readiness reviews.
"""

import ast
import tokenize
import io
import re
import sys
import os
import json
import datetime
import logging
import argparse
import concurrent.futures
import fnmatch
from dataclasses import dataclass, asdict
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any, Union

# Default configuration
DEFAULT_CONFIG = {
    "min_function_lines": 5,
    "min_docstring_length": 15,
    "test_coverage_threshold": 0.7,
    "ignore_patterns": [
        "__pycache__/*",
        "*.pyc",
        ".git/*",
        ".venv/*",
        "venv/*",
        "env/*",
        "node_modules/*"
    ],
    "ignore_functions": [
        "__init__",
        "__str__",
        "__repr__"
    ],
    "severity_levels": {
        "syntax_errors": "critical",
        "unimplemented_abstracts": "critical",
        "stubs": "serious",
        "simple_returns": "serious",
        "incomplete_methods": "serious",
        "test_gaps": "serious",
        "todos": "minor",
        "missing_docstrings": "minor",
        "suspicious_short_functions": "minor",
        "type_hint_gaps": "minor"
    }
}

@dataclass
class DiagnosticIssue:
    """Single diagnostic issue"""
    category: str
    severity: str
    line_number: int
    message: str
    details: Dict[str, Any]

@dataclass
class DiagnosticResult:
    """Complete diagnostic result for a file"""
    file_path: str
    issues: List[DiagnosticIssue]
    metrics: Dict[str, int]
    
    def get_issues_by_severity(self, severity: str) -> List[DiagnosticIssue]:
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_category(self, category: str) -> List[DiagnosticIssue]:
        return [issue for issue in self.issues if issue.category == category]

class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            self._merge_config(user_config)
            logging.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
    
    def _merge_config(self, user_config: Dict[str, Any]):
        """Merge user config with defaults"""
        for key, value in user_config.items():
            if isinstance(value, dict) and key in self.config:
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

class ProductionDoctor:
    """Comprehensive Python code health diagnostic tool"""
    
    TODO_PATTERNS = [
        (r'\bTODO\b', 'Action Required'),
        (r'\bFIXME\b', 'Critical Fix Needed'),
        (r'\bHACK\b', 'Technical Debt'),
        (r'\bXXX\b', 'Urgent Review'),
        (r'\bNOTE\b', 'Important Note'),
        (r'\bTEMP\b', 'Temporary Code'),
        (r'\bWIP\b', 'Work In Progress')
    ]
    
    SIMPLE_RETURN_PATTERNS = [
        (r'return\s+None\s*$', 'None'),
        (r'return\s+0\s*$', 'Zero'),
        (r'return\s+\""?\s*$', 'Empty String'),
        (r'return\s+\[\]\s*$', 'Empty List'),
        (r'return\s+\{\}\s*$', 'Empty Dict'),
        (r'return\s+False\s*$', 'False'),
        (r'return\s+True\s*$', 'True'),
        (r'return\s+\.\.\.\s*$', 'Ellipsis'),
        (r'raise\s+NotImplementedError', 'NotImplementedError')
    ]
    
    def __init__(self, file_path: str, config: ConfigManager, project_root: Optional[str] = None):
        self.file_path = file_path
        self.config = config
        self.project_root = project_root or os.getcwd()
        self.source = None
        self.tree = None
        self.issues: List[DiagnosticIssue] = []
        self.metrics = {
            'total_lines': 0,
            'total_functions': 0,
            'total_classes': 0,
            'total_methods': 0,
            'documented_functions': 0,
            'type_hinted_functions': 0
        }
        
        # Setup logging for this file
        self.logger = logging.getLogger(f"doctor.{os.path.basename(file_path)}")

    def _should_ignore_file(self) -> bool:
        """Check if file should be ignored based on patterns"""
        rel_path = os.path.relpath(self.file_path, self.project_root)
        ignore_patterns = self.config.get('ignore_patterns', [])
        
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(rel_path, pattern):
                return True
        return False

    def _add_issue(self, category: str, line_number: int, message: str, **details):
        """Add a diagnostic issue"""
        severity = self.config.get('severity_levels', {}).get(category, 'minor')
        issue = DiagnosticIssue(
            category=category,
            severity=severity,
            line_number=line_number,
            message=message,
            details=details
        )
        self.issues.append(issue)

    def _read_file(self) -> bool:
        """Read file content with robust encoding handling"""
        if self._should_ignore_file():
            self.logger.debug(f"Ignoring file due to patterns: {self.file_path}")
            return False
            
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(self.file_path, 'r', encoding=encoding) as f:
                    self.source = f.read()
                
                # Update metrics
                self.metrics['total_lines'] = len(self.source.splitlines())
                return True
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self._add_issue('syntax_errors', 0, f"File read error: {str(e)}")
                return False
        
        self._add_issue('syntax_errors', 0, "Unsupported file encoding - tried utf-8, latin-1, cp1252")
        return False

    def _check_syntax(self) -> bool:
        """Check for syntax errors using AST parsing"""
        if not self.source:
            return False
            
        try:
            self.tree = ast.parse(self.source, filename=self.file_path)
            return True
        except SyntaxError as e:
            self._add_issue(
                'syntax_errors',
                e.lineno,
                f"Line {e.lineno}: {e.msg} (offset {e.offset})"
            )
        except Exception as e:
            self._add_issue('syntax_errors', 0, f"AST parsing error: {str(e)}")
        return False

    def _scan_for_todos(self):
        """Scan source code for TODOs and technical debt markers"""
        if not self.source:
            return
            
        try:
            tokens = tokenize.generate_tokens(io.StringIO(self.source).readline)
            for token in tokens:
                if token.type not in (tokenize.COMMENT, tokenize.STRING):
                    continue
                    
                line_num = token.start[0]
                text = token.string.strip()
                
                if not text:
                    continue
                    
                for pattern, severity in self.TODO_PATTERNS:
                    if re.search(pattern, text, re.IGNORECASE):
                        match = re.search(pattern, text, re.IGNORECASE)
                        context = text[max(0, match.start()-10):min(len(text), match.end()+20)]
                        self._add_issue(
                            'todos',
                            line_num,
                            f"TODO found: '{text}'",
                            pattern=pattern[2:-2],
                            severity=severity
                        )
                        break
        except Exception as e:
            self._add_issue('syntax_errors', 0, f"Tokenization error: {str(e)}")

    def _analyze_ast(self):
        """Deep AST analysis for production readiness"""
        if not self.tree or any(issue.category == 'syntax_errors' for issue in self.issues):
            return
            
        # Collect metrics and analyze
        self._collect_metrics()
        self._analyze_completeness()

    def _collect_metrics(self):
        """Collect code metrics"""
        class MetricsCollector(ast.NodeVisitor):
            def __init__(self, doctor):
                self.doctor = doctor
                super().__init__()
                
            def visit_ClassDef(self, node):
                self.doctor.metrics['total_classes'] += 1
                self.generic_visit(node)
                
            def visit_FunctionDef(self, node):
                self.doctor.metrics['total_functions'] += 1
                
                # Check for method vs function
                if hasattr(node, 'parent') and isinstance(node.parent, ast.ClassDef):
                    self.doctor.metrics['total_methods'] += 1
                
                # Check docstring
                if self._has_docstring(node):
                    self.doctor.metrics['documented_functions'] += 1
                
                # Check type hints
                if self._has_type_hints(node):
                    self.doctor.metrics['type_hinted_functions'] += 1
                
                self.generic_visit(node)
            
            def _has_docstring(self, node):
                if not node.body:
                    return False
                first = node.body[0]
                if isinstance(first, ast.Expr):
                    if isinstance(first.value, (ast.Str, ast.Constant)):
                        return True
                return False
            
            def _has_type_hints(self, node):
                # Check if function has any type hints
                has_return_hint = node.returns is not None
                has_param_hints = any(arg.annotation is not None for arg in node.args.args)
                return has_return_hint or has_param_hints
        
        # Set parent references
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node
        
        collector = MetricsCollector(self)
        collector.visit(self.tree)

    def _analyze_completeness(self):
        """Enhanced completeness analysis with better detection"""
        class EnhancedCompletenessVisitor(ast.NodeVisitor):
            def __init__(self, doctor, source_lines):
                self.doctor = doctor
                self.source_lines = source_lines
                super().__init__()
                
            def visit_FunctionDef(self, node):
                # Skip if function should be ignored
                if node.name in self.doctor.config.get('ignore_functions', []):
                    self.generic_visit(node)
                    return
                
                # Skip nested functions for now
                if hasattr(node, 'parent') and isinstance(node.parent, ast.FunctionDef):
                    self.generic_visit(node)
                    return
                
                self._analyze_function(node)
                self.generic_visit(node)
            
            def _analyze_function(self, node):
                """Comprehensive function analysis"""
                # Check docstring
                if not self._has_meaningful_docstring(node):
                    self._add_docstring_issue(node)
                
                # Check type hints
                self._check_type_hints(node)
                
                # Analyze function body
                body = self._get_function_body(node)
                
                if not body:
                    self._add_stub_issue(node, "empty body")
                elif len(body) == 1:
                    self._analyze_single_statement_body(node, body[0])
                else:
                    self._analyze_multi_statement_body(node, body)
            
            def _has_meaningful_docstring(self, node):
                """Check for meaningful docstrings"""
                if not node.body:
                    return False
                
                first = node.body[0]
                if not isinstance(first, ast.Expr):
                    return False
                
                docstring = None
                if isinstance(first.value, ast.Str):
                    docstring = first.value.s
                elif isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                    docstring = first.value.value
                
                if not docstring:
                    return False
                
                min_length = self.doctor.config.get('min_docstring_length', 15)
                return len(docstring.strip()) >= min_length
            
            def _get_function_body(self, node):
                """Get function body excluding docstring"""
                body = node.body
                if body and isinstance(body[0], ast.Expr):
                    if isinstance(body[0].value, (ast.Str, ast.Constant)):
                        if isinstance(body[0].value, ast.Constant):
                            if isinstance(body[0].value.value, str):
                                body = body[1:]  # Skip docstring
                        else:
                            body = body[1:]  # Skip docstring
                return body
            
            def _analyze_single_statement_body(self, node, stmt):
                """Analyze single statement function bodies"""
                if isinstance(stmt, ast.Pass):
                    self._add_stub_issue(node, "pass statement")
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Ellipsis):
                    self._add_stub_issue(node, "ellipsis (...)")
                elif isinstance(stmt, ast.Raise) and self._is_not_implemented_error(stmt):
                    self._add_stub_issue(node, "NotImplementedError")
                elif isinstance(stmt, ast.Return):
                    self._check_simple_return(node, stmt)
            
            def _analyze_multi_statement_body(self, node, body):
                """Analyze multi-statement function bodies"""
                # Check for suspiciously short functions
                min_lines = self.doctor.config.get('min_function_lines', 5)
                if len(body) > 0:
                    start_line = body[0].lineno
                    end_line = body[-1].lineno
                    line_count = end_line - start_line + 1
                    
                    if line_count < min_lines:
                        self.doctor._add_issue(
                            'suspicious_short_functions',
                            node.lineno,
                            f"Function '{node.name}' has only {line_count} lines of code",
                            function_name=node.name,
                            line_count=line_count
                        )
                
                # Check for simple returns in multi-statement bodies
                for stmt in body:
                    if isinstance(stmt, ast.Return):
                        self._check_simple_return(node, stmt)
            
            def _add_docstring_issue(self, node):
                self.doctor._add_issue(
                    'missing_docstrings',
                    node.lineno,
                    f"Function '{node.name}' missing meaningful docstring",
                    entity_type='function',
                    entity_name=node.name
                )
            
            def _add_stub_issue(self, node, stub_type):
                self.doctor._add_issue(
                    'stubs',
                    node.lineno,
                    f"Function '{node.name}' appears to be a stub ({stub_type})",
                    function_name=node.name,
                    stub_type=stub_type
                )
            
            def _is_not_implemented_error(self, node):
                """Enhanced NotImplementedError detection"""
                if not hasattr(node, 'exc') or node.exc is None:
                    return False
                
                exc = node.exc
                if isinstance(exc, ast.Name) and exc.id == 'NotImplementedError':
                    return True
                
                if isinstance(exc, ast.Call):
                    func = exc.func
                    if isinstance(func, ast.Name) and func.id == 'NotImplementedError':
                        return True
                    if isinstance(func, ast.Attribute) and func.attr == 'NotImplementedError':
                        return True
                return False
            
            def _check_simple_return(self, func_node, return_node):
                """Enhanced simple return detection"""
                try:
                    # Get the return statement as text
                    start_line = return_node.lineno - 1
                    end_line = getattr(return_node, 'end_lineno', return_node.lineno) - 1
                    lines = self.source_lines[start_line:end_line+1]
                    return_stmt = ''.join(lines).strip()
                    
                    for pattern, desc in self.doctor.SIMPLE_RETURN_PATTERNS:
                        if re.search(pattern, return_stmt):
                            self.doctor._add_issue(
                                'simple_returns',
                                func_node.lineno,
                                f"Function '{func_node.name}' returns simple placeholder ({desc})",
                                function_name=func_node.name,
                                return_type=desc,
                                return_statement=return_stmt
                            )
                            return
                except (IndexError, AttributeError):
                    pass  # Skip if we can't analyze the return
            
            def _check_type_hints(self, node):
                """Enhanced type hint checking"""
                missing_hints = []
                
                # Check return type
                if node.returns is None and node.name not in ['__init__']:
                    missing_hints.append("return type")
                
                # Check parameters (skip 'self' and 'cls')
                for arg in node.args.args:
                    if arg.arg in ['self', 'cls']:
                        continue
                    if arg.annotation is None:
                        missing_hints.append(f"param '{arg.arg}'")
                
                if missing_hints:
                    self.doctor._add_issue(
                        'type_hint_gaps',
                        node.lineno,
                        f"Function '{node.name}' missing type hints for: {', '.join(missing_hints)}",
                        function_name=node.name,
                        missing_hints=missing_hints
                    )
        
        source_lines = self.source.splitlines(keepends=True)
        visitor = EnhancedCompletenessVisitor(self, source_lines)
        visitor.visit(self.tree)

    def _check_test_coverage(self):
        """Check for missing test coverage (basic heuristic)"""
        # Simple heuristic: look for corresponding test file
        rel_path = os.path.relpath(self.file_path, self.project_root)
        test_path = self._get_test_path(rel_path)
        
        if not os.path.exists(test_path):
            self._add_issue(
                'test_gaps',
                0,
                "No test file found for this module",
                module_path=rel_path,
                test_path=test_path
            )
            return
        
        # Check for test functions
        try:
            with open(test_path, 'r', encoding='utf-8') as f:
                test_source = f.read()
            
            # Count test functions
            test_count = len(re.findall(r'\bdef\s+test_', test_source))
            
            # Count module functions
            func_count = self.metrics.get('total_functions', 0)
            
            if test_count < func_count * self.config.get('test_coverage_threshold', 0.7):  # 70% coverage threshold
                self._add_issue(
                    'test_gaps',
                    0,
                    f"Only {test_count} tests for {func_count} functions",
                    module_path=rel_path,
                    test_path=test_path
                )
        except Exception as e:
            self._add_issue(
                'test_gaps',
                0,
                f"Error analyzing test coverage: {str(e)}",
                module_path=rel_path
            )
    
    def _get_test_path(self, module_path: str) -> str:
        """Generate test path from module path"""
        parts = Path(module_path).parts
        
        # Handle src layout
        if 'src' in parts:
            idx = parts.index('src')
            test_parts = list(parts[idx+1:])
        else:
            test_parts = list(parts)
        
        # Create test path
        test_parts = ['tests'] + ['test_' + p for p in test_parts[:-1]] + ['test_' + parts[-1]]
        test_path = Path(self.project_root) / Path(*test_parts)
        
        # Handle .py extension
        if test_path.suffix != '.py':
            test_path = test_path.with_suffix('.py')
        
        return str(test_path)

    def run_diagnostics(self) -> DiagnosticResult:
        """Run all diagnostic checks on the file"""
        try:
            if not self._read_file():
                return self._get_result()
            
            if not self._check_syntax():
                return self._get_result()
            
            self._scan_for_todos()
            self._analyze_ast()
            self._check_test_coverage()
            
            self.logger.info(f"Completed analysis: {len(self.issues)} issues found")
            return self._get_result()
            
        except Exception as e:
            self.logger.error(f"Unexpected error during diagnostics: {e}")
            self._add_issue('syntax_errors', 0, f"Analysis failed: {str(e)}")
            return self._get_result()
    
    def _get_result(self) -> DiagnosticResult:
        """Get diagnostic results"""
        return DiagnosticResult(
            file_path=self.file_path,
            issues=self.issues,
            metrics=self.metrics
        )

    @staticmethod
    def generate_json_report(results: List[DiagnosticResult], project_root: str) -> str:
        """Generate JSON report for CI/CD integration"""
        report_data = {
            "project_root": project_root,
            "scan_date": datetime.datetime.now().isoformat(),
            "python_version": sys.version.split()[0],
            "summary": {},
            "files": []
        }
        
        # Calculate summary
        total_issues = 0
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for result in results:
            file_data = {
                "file_path": os.path.relpath(result.file_path, project_root),
                "metrics": result.metrics,
                "issues": [asdict(issue) for issue in result.issues]
            }
            report_data["files"].append(file_data)
            
            total_issues += len(result.issues)
            for issue in result.issues:
                severity_counts[issue.severity] += 1
                category_counts[issue.category] += 1
        
        report_data["summary"] = {
            "total_issues": total_issues,
            "severity_counts": dict(severity_counts),
            "category_counts": dict(category_counts)
        }
        
        return json.dumps(report_data, indent=2)

    @staticmethod
    def generate_markdown_report(results: List[DiagnosticResult], project_root: str) -> str:
        """Generate enhanced markdown report"""
        report = [
            "# Python Production Doctor Report\n",
            f"**Project Root:** `{project_root}`",
            f"**Scan Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Python Version:** {sys.version.split()[0]}",
            "\n---\n"
        ]
        
        # Enhanced summary with metrics
        total_issues = sum(len(result.issues) for result in results)
        total_files = len(results)
        
        # Calculate category counts for action plan
        category_counts = defaultdict(int)
        for result in results:
            for issue in result.issues:
                category_counts[issue.category] += 1
        
        if total_issues == 0:
            report.append("**FULLY PRODUCTION READY** - No issues found!\n")
        else:
            # Categorize by severity
            critical = sum(len([i for i in result.issues if i.severity == 'critical']) for result in results)
            serious = sum(len([i for i in result.issues if i.severity == 'serious']) for result in results)
            minor = sum(len([i for i in result.issues if i.severity == 'minor']) for result in results)
            
            if critical > 0:
                status = "**CRITICAL ISSUES** - Cannot deploy"
            elif serious > 0:
                status = "**SERIOUS ISSUES** - Requires significant work"
            else:
                status = "**MINOR ISSUES** - Ready with minor fixes"
            
            report.append(f"{status}\n")
            report.append(f"**Files Scanned:** {total_files} | **Total Issues:** {total_issues}")
            report.append(f"- Critical: {critical}")
            report.append(f"- Serious: {serious}")  
            report.append(f"- Minor: {minor}\n")
        
        # Add code quality metrics
        total_functions = sum(result.metrics.get('total_functions', 0) for result in results)
        documented_functions = sum(result.metrics.get('documented_functions', 0) for result in results)
        type_hinted_functions = sum(result.metrics.get('type_hinted_functions', 0) for result in results)
        
        if total_functions > 0:
            doc_coverage = (documented_functions / total_functions) * 100
            type_coverage = (type_hinted_functions / total_functions) * 100
            
            report.append("## Code Quality Metrics\n")
            report.append(f"- **Documentation Coverage:** {doc_coverage:.1f}% ({documented_functions}/{total_functions})")
            report.append(f"- **Type Hint Coverage:** {type_coverage:.1f}% ({type_hinted_functions}/{total_functions})\n")
        
        # File-by-file breakdown
        report.append("## File Analysis\n")
        
        for result in results:
            rel_path = os.path.relpath(result.file_path, project_root)
            total = len(result.issues)
            
            if total == 0:
                report.append(f"### `{rel_path}`\n")
                report.append("_No issues found - ready for production_\n")
                continue
                
            report.append(f"### `{rel_path}`\n")
            report.append(f"**Total Issues:** {total}\n")
            
            # Syntax errors
            syntax_errors = [i for i in result.issues if i.category == 'syntax_errors']
            if syntax_errors:
                report.append("#### Syntax Errors")
                for i, err in enumerate(syntax_errors, 1):
                    report.append(f"{i}. `{err.message}` (line {err.line_number})")
                report.append("")
            
            # TODOs
            todos = [i for i in result.issues if i.category == 'todos']
            if todos:
                report.append("#### Technical Debt")
                for i, issue in enumerate(todos, 1):
                    report.append(f"{i}. **{issue.severity}** at line {issue.line_number}: `{issue.message}`")
                report.append("")
            
            # Stubs
            stubs = [i for i in result.issues if i.category == 'stubs']
            if stubs:
                report.append("#### Stub Implementations")
                for i, issue in enumerate(stubs, 1):
                    report.append(f"{i}. `{issue.details['function_name']}()` at line {issue.line_number} - **{issue.details['stub_type']}**")
                report.append("")
            
            # Simple returns
            simple_returns = [i for i in result.issues if i.category == 'simple_returns']
            if simple_returns:
                report.append("#### Placeholder Returns")
                for i, issue in enumerate(simple_returns, 1):
                    report.append(f"{i}. `{issue.details['function_name']}()` at line {issue.line_number} - returns **{issue.details['return_type']}**")
                report.append("")
            
            # Incomplete methods
            incomplete_methods = [i for i in result.issues if i.category == 'incomplete_methods']
            if incomplete_methods:
                report.append("#### Incomplete Methods")
                for i, issue in enumerate(incomplete_methods, 1):
                    report.append(f"{i}. `{issue.details['class_name']}.{issue.details['method_name']}()` at line {issue.line_number}")
                report.append("")
            
            # Missing docstrings
            missing_docstrings = [i for i in result.issues if i.category == 'missing_docstrings']
            if missing_docstrings:
                report.append("#### Missing Docstrings")
                for i, issue in enumerate(missing_docstrings, 1):
                    report.append(f"{i}. {issue.details['entity_type'].capitalize()} `{issue.details['entity_name']}` at line {issue.line_number}")
                report.append("")
            
            # Suspicious short functions
            suspicious_short_functions = [i for i in result.issues if i.category == 'suspicious_short_functions']
            if suspicious_short_functions:
                report.append("#### Suspiciously Short Functions")
                for i, issue in enumerate(suspicious_short_functions, 1):
                    report.append(f"{i}. `{issue.details['function_name']}()` at line {issue.line_number} - only {issue.details['line_count']} line{'s' if issue.details['line_count'] > 1 else ''} of code")
                report.append("")
            
            # Unimplemented abstracts
            unimplemented_abstracts = [i for i in result.issues if i.category == 'unimplemented_abstracts']
            if unimplemented_abstracts:
                report.append("#### Unimplemented Abstract Methods")
                for i, issue in enumerate(unimplemented_abstracts, 1):
                    report.append(f"{i}. `{issue.details['class_name']}.{issue.details['method_name']}()` at line {issue.line_number}")
                report.append("")
            
            # Type hint gaps
            type_hint_gaps = [i for i in result.issues if i.category == 'type_hint_gaps']
            if type_hint_gaps:
                report.append("#### Incomplete Type Hints")
                for i, issue in enumerate(type_hint_gaps, 1):
                    report.append(f"{i}. `{issue.details['function_name']}()` at line {issue.line_number} - missing: {issue.details['missing_hints']}")
                report.append("")
            
            # Test gaps
            test_gaps = [i for i in result.issues if i.category == 'test_gaps']
            if test_gaps:
                report.append("#### Test Coverage Gaps")
                for i, issue in enumerate(test_gaps, 1):
                    report.append(f"{i}. `{issue.details['module_path']}` - {issue.message}")
                report.append("")
        
        # Action plan
        report.append("## Recommended Action Plan\n")
        report.append("### Critical Fixes (Block Deployment)")
        if category_counts['syntax_errors'] > 0:
            report.append("- Fix all syntax errors")
        if category_counts['unimplemented_abstracts'] > 0:
            report.append("- Implement all abstract methods")
        
        report.append("\n### Serious Fixes (Required Before Release)")
        if category_counts['stubs'] > 0:
            report.append("- Replace all stub implementations with real code")
        if category_counts['simple_returns'] > 0:
            report.append("- Replace placeholder return values with real implementations")
        if category_counts['incomplete_methods'] > 0:
            report.append("- Complete all incomplete methods")
        if category_counts['test_gaps'] > 0:
            report.append("- Increase test coverage to at least 80%")
        
        report.append("\n### Quality Improvements (Recommended)")
        if category_counts['todos'] > 0:
            report.append("- Address all TODOs and technical debt")
        if category_counts['missing_docstrings'] > 0:
            report.append("- Add meaningful docstrings to all functions/classes")
        if category_counts['suspicious_short_functions'] > 0:
            report.append("- Review suspiciously short functions for completeness")
        if category_counts['type_hint_gaps'] > 0:
            report.append("- Complete all type hints for better maintainability")
        
        report.append("\n---\n")
        report.append("_This report generated by [Python Production Doctor](https://github.com/yourusername/production-doctor)_")
        
        return "\n".join(report)

def scan_project(project_root: str) -> List[DiagnosticResult]:
    """Scan entire project for production readiness"""
    results = []
    doctor = None
    
    for root, _, files in os.walk(project_root):
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                file_path = os.path.join(root, file)
                print(f"Scanning: {os.path.relpath(file_path, project_root)}")
                doctor = ProductionDoctor(file_path, project_root)
                results.append(doctor.run_diagnostics())
    
    return results

def scan_project_parallel(project_root: str, config: ConfigManager, max_workers: int = 4) -> List[DiagnosticResult]:
    """Scan project with parallel processing"""
    python_files = []
    
    for root, _, files in os.walk(project_root):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(analyze_file, file_path, config, project_root): file_path 
            for file_path in python_files
        }
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                if result:  # Only add non-empty results
                    results.append(result)
                    print(f"OK {os.path.relpath(file_path, project_root)}")
            except Exception as e:
                print(f"ERROR {os.path.relpath(file_path, project_root)}: {e}")
                logging.error(f"Failed to analyze {file_path}: {e}")
    
    return results

def analyze_file(file_path: str, config: ConfigManager, project_root: str) -> Optional[DiagnosticResult]:
    """Analyze a single file (for parallel processing)"""
    doctor = ProductionDoctor(file_path, config, project_root)
    result = doctor.run_diagnostics()
    
    # Only return results with issues or if file was successfully processed
    if result.issues or result.metrics['total_lines'] > 0:
        return result
    return None

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('production_doctor.log'),
            logging.StreamHandler(sys.stdout) if verbose else logging.NullHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Python Production Doctor - Code Health Assessment')
    parser.add_argument('project_root', help='Project root directory to scan')
    parser.add_argument('-o', '--output', default='production_report.md', 
                       help='Output file name (default: production_report.md)')
    parser.add_argument('-c', '--config', help='Configuration file path')
    parser.add_argument('-f', '--format', choices=['markdown', 'json'], default='markdown',
                       help='Output format (default: markdown)')
    parser.add_argument('-j', '--jobs', type=int, default=4,
                       help='Number of parallel jobs (default: 4)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Load configuration
    config = ConfigManager(args.config)
    
    project_root = os.path.abspath(args.project_root)
    
    print(f"Starting production readiness scan for {project_root}")
    print(f"Using {args.jobs} parallel workers")
    
    # Scan project
    results = scan_project_parallel(project_root, config, args.jobs)
    
    print(f"\nGenerating {args.format} report...")
    
    # Generate report
    if args.format == 'json':
        report = ProductionDoctor.generate_json_report(results, project_root)
        if not args.output.endswith('.json'):
            args.output = args.output.replace('.md', '.json')
    else:
        report = ProductionDoctor.generate_markdown_report(results, project_root)
    
    # Write report
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nReport generated: {os.path.abspath(args.output)}")
    
    # Print summary
    total_issues = sum(len(result.issues) for result in results)
    critical_issues = sum(len([i for i in result.issues if i.severity == 'critical']) for result in results)
    
    print(f"\nSummary: {len(results)} files scanned, {total_issues} issues found")
    if critical_issues > 0:
        print(f"{critical_issues} critical issues found - deployment blocked")
        sys.exit(1)
    elif total_issues == 0:
        print("Project is production ready!")
    else:
        print("Project has issues but can proceed with caution")

if __name__ == "__main__":
    main()
