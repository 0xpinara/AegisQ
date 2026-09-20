"""A parser for the subset of OpenQASM that AegisQ can execute.

This is **not** a complete OpenQASM implementation and does not pretend to be.
It accepts the declarations and gate statements that map directly onto the
AegisQ instruction set, and rejects everything else with a line number and a
reason rather than silently ignoring it — a silently dropped instruction would
produce a wrong state and a plausible-looking benchmark.

Supported grammar
-----------------

```
OPENQASM 2.0;  |  OPENQASM 3;          // optional version header
include "qelib1.inc";                  // accepted and ignored
qreg q[5];     |  qubit[5] q;          // one quantum register
creg c[5];     |  bit[5] c;            // accepted and ignored
x q[0];                                // x y z h s t
rx(0.5) q[1];                          // rx ry rz, one real parameter
cx q[0], q[1];                         // cx cz swap
measure q[0] -> c[0];  |  c[0] = measure q[0];  |  measure q[0];
barrier q;                             // accepted and ignored
// line comments
```

Parameters may be numeric literals or simple `pi` expressions such as
`pi/4`, `2*pi`, `-pi/8`.

Not supported: custom `gate` definitions, classical control, `if`, loops,
multiple registers, register-wide gate application, and every gate outside the
AegisQ set. Each raises `QasmError`.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from aegisq.circuit.circuit import Circuit
from aegisq.circuit.gates import GATE_SPECS
from aegisq.circuit.validation import CircuitError


class QasmError(CircuitError):
    """Raised when a QASM source uses something outside the supported subset."""

    def __init__(self, message: str, line_number: int | None = None, line: str | None = None):
        if line_number is not None:
            message = f"line {line_number}: {message}"
            if line:
                message = f"{message}\n    {line.strip()}"
        super().__init__(message)


#: `gate q[i];`, `gate(param) q[i];`, `gate q[i], q[j];`
_STATEMENT = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\((?P<params>[^)]*)\))?"
    r"\s+(?P<operands>.+)$"
)
_OPERAND = re.compile(r"^(?P<reg>[A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(?P<index>\d+)\s*\]$")
_QREG = re.compile(
    r"^(?:qreg\s+(?P<name1>[A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(?P<size1>\d+)\s*\]"
    r"|qubit\s*\[\s*(?P<size2>\d+)\s*\]\s*(?P<name2>[A-Za-z_][A-Za-z0-9_]*))$"
)
_CREG = re.compile(r"^(?:creg\s+\w+\s*\[\s*\d+\s*\]|bit\s*\[\s*\d+\s*\]\s*\w+)$")
_MEASURE_ARROW = re.compile(r"^measure\s+(?P<source>.+?)\s*->\s*(?P<target>.+)$")
_MEASURE_ASSIGN = re.compile(r"^(?P<target>.+?)\s*=\s*measure\s+(?P<source>.+)$")
_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _parse_parameter(text: str, line_number: int, line: str) -> float:
    """Evaluate a numeric literal or a simple multiple/fraction of pi."""
    token = text.strip().replace(" ", "")
    if not token:
        raise QasmError("empty gate parameter", line_number, line)
    if _NUMBER.match(token):
        return float(token)

    # Accept the common pi forms without evaluating arbitrary expressions.
    sign = 1.0
    while token and token[0] in "+-":
        if token[0] == "-":
            sign = -sign
        token = token[1:]
    if token == "pi":
        return sign * math.pi
    fraction = re.match(r"^pi/(\d+\.?\d*)$", token)
    if fraction:
        return sign * math.pi / float(fraction.group(1))
    multiple = re.match(r"^(\d+\.?\d*)\*pi$", token)
    if multiple:
        return sign * float(multiple.group(1)) * math.pi
    multiple_fraction = re.match(r"^(\d+\.?\d*)\*pi/(\d+\.?\d*)$", token)
    if multiple_fraction:
        return (
            sign * float(multiple_fraction.group(1)) * math.pi / float(multiple_fraction.group(2))
        )
    raise QasmError(
        f"unsupported parameter expression {text.strip()!r}; "
        "use a number or a simple pi expression such as pi/4",
        line_number,
        line,
    )


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return "\n".join(line.split("//", 1)[0] for line in source.splitlines())


def parse_qasm(source: str, name: str = "qasm") -> Circuit:
    """Parse OpenQASM source into a :class:`Circuit`."""
    cleaned = _strip_comments(source)

    register_name: str | None = None
    register_size: int | None = None
    circuit: Circuit | None = None
    pending: list[tuple[int, str]] = []

    # Statements are separated by semicolons but may span lines; track the
    # physical line number of each statement for error messages.
    line_number = 0
    buffer = ""
    statements: list[tuple[int, str]] = []
    for raw_line in cleaned.splitlines():
        line_number += 1
        buffer += " " + raw_line
        while ";" in buffer:
            statement, buffer = buffer.split(";", 1)
            statement = statement.strip()
            if statement:
                statements.append((line_number, statement))
    if buffer.strip():
        # A leftover block delimiter means the source uses a construct with a
        # body, which is more useful to name than "missing semicolon".
        if "{" in buffer or "}" in buffer:
            raise QasmError(
                "custom gate definitions, blocks and classical control are not supported",
                line_number,
                buffer,
            )
        raise QasmError("statement is missing a terminating ';'", line_number, buffer)

    for number, statement in statements:
        lowered = statement.lower()

        if lowered.startswith("openqasm"):
            continue
        if lowered.startswith("include"):
            continue
        if lowered.startswith("barrier"):
            continue

        register_match = _QREG.match(statement)
        if register_match:
            if register_name is not None:
                raise QasmError(
                    "AegisQ supports a single quantum register per circuit", number, statement
                )
            register_name = register_match.group("name1") or register_match.group("name2")
            register_size = int(register_match.group("size1") or register_match.group("size2"))
            circuit = Circuit(register_size, name=name)
            continue

        if _CREG.match(statement):
            continue

        if circuit is None:
            raise QasmError(
                "gate statement before any quantum register declaration", number, statement
            )

        measure_match = _MEASURE_ARROW.match(statement) or _MEASURE_ASSIGN.match(statement)
        if measure_match:
            source_operand = measure_match.group("source").strip()
            circuit.measure(
                _operand_index(source_operand, register_name, register_size, number, statement)
            )
            continue
        if lowered.startswith("measure "):
            circuit.measure(
                _operand_index(
                    statement[len("measure ") :].strip(),
                    register_name,
                    register_size,
                    number,
                    statement,
                )
            )
            continue
        if lowered.startswith("reset"):
            raise QasmError("reset is not supported", number, statement)
        if (
            lowered.startswith("if")
            or lowered.startswith("gate ")
            or lowered.startswith("for ")
            or lowered.startswith("while ")
            or "{" in statement
        ):
            raise QasmError(
                "custom gate definitions, blocks and classical control are not supported",
                number,
                statement,
            )

        pending.append((number, statement))
        _apply_gate_statement(circuit, register_name, register_size, number, statement)

    if circuit is None:
        raise QasmError("no quantum register declared")
    return circuit


def _operand_index(
    operand: str, register: str | None, size: int | None, number: int, statement: str
) -> int:
    text = operand.strip()
    match = _OPERAND.match(text)
    if not match:
        # Distinguish the two ways this goes wrong: a whole register was named,
        # or the index itself is malformed.
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
            raise QasmError(
                f"operand {text!r} names a whole register; AegisQ requires an "
                f"indexed qubit such as {text}[0]",
                number,
                statement,
            )
        raise QasmError(
            f"operand {text!r} is malformed; expected {register or 'q'}[i] with a "
            "non-negative integer index",
            number,
            statement,
        )
    if register is not None and match.group("reg") != register:
        raise QasmError(
            f"unknown register {match.group('reg')!r}; declared register is {register!r}",
            number,
            statement,
        )
    index = int(match.group("index"))
    if size is not None and not 0 <= index < size:
        raise QasmError(
            f"qubit index {index} outside the declared register of size {size}", number, statement
        )
    return index


def _apply_gate_statement(
    circuit: Circuit, register: str | None, size: int | None, number: int, statement: str
) -> None:
    match = _STATEMENT.match(statement)
    if not match:
        raise QasmError(f"cannot parse statement {statement!r}", number, statement)

    opcode = match.group("name")
    if opcode not in GATE_SPECS:
        # OpenQASM identifiers are case-sensitive, and so are the declaration
        # keywords this parser already rejects when capitalised. Accepting
        # `H` while rejecting `QREG` would be an inconsistency dressed up as
        # convenience, so the case is called out instead of silently fixed.
        if opcode.lower() in GATE_SPECS:
            raise QasmError(
                f"gate names are case-sensitive; write {opcode.lower()!r} rather than {opcode!r}",
                number,
                statement,
            )
        raise QasmError(
            f"gate {opcode!r} is outside the AegisQ instruction set "
            f"({', '.join(sorted(GATE_SPECS))})",
            number,
            statement,
        )

    spec = GATE_SPECS[opcode]
    raw_params = match.group("params")
    params = (
        [_parse_parameter(part, number, statement) for part in raw_params.split(",")]
        if raw_params and raw_params.strip()
        else []
    )
    if len(params) != spec.num_params:
        raise QasmError(
            f"{opcode} takes {spec.num_params} parameter(s), got {len(params)}", number, statement
        )

    operands = [op for op in match.group("operands").split(",") if op.strip()]
    if len(operands) != spec.num_qubits:
        raise QasmError(
            f"{opcode} acts on {spec.num_qubits} qubit(s), got {len(operands)}", number, statement
        )
    qubits = [_operand_index(op, register, size, number, statement) for op in operands]

    try:
        if spec.num_qubits == 1:
            getattr(circuit, opcode)(qubits[0], *params)
        else:
            getattr(circuit, opcode)(qubits[0], qubits[1])
    except CircuitError as exc:
        raise QasmError(str(exc), number, statement) from None


def parse_qasm_file(path: str | Path) -> Circuit:
    """Parse a `.qasm` file, naming the circuit after the file."""
    path = Path(path)
    return parse_qasm(path.read_text(encoding="utf-8"), name=path.stem)


def to_qasm(circuit: Circuit, version: str = "2.0") -> str:
    """Emit the circuit as OpenQASM in the same supported subset."""
    if version not in ("2.0", "3"):
        raise ValueError("version must be '2.0' or '3'")
    lines = []
    if version == "2.0":
        lines += ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{circuit.num_qubits}];"]
        if circuit.measured_qubits:
            lines.append(f"creg c[{circuit.num_qubits}];")
    else:
        lines += ["OPENQASM 3;", f"qubit[{circuit.num_qubits}] q;"]
        if circuit.measured_qubits:
            lines.append(f"bit[{circuit.num_qubits}] c;")

    for gate in circuit:
        operands = ", ".join(f"q[{q}]" for q in gate.qubits)
        if gate.params:
            params = ", ".join(f"{p!r}" for p in gate.params)
            lines.append(f"{gate.opcode}({params}) {operands};")
        else:
            lines.append(f"{gate.opcode} {operands};")
    for q in circuit.measured_qubits:
        lines.append(f"measure q[{q}] -> c[{q}];")
    return "\n".join(lines) + "\n"
