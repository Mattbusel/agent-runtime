//! Structured inter-agent communication protocol.
//!
//! Provides typed message headers, CRC-32-verified payloads, sequence tracking,
//! and encode/decode round-trips for all inter-agent messages.

use std::fmt;

// ── MessageType ───────────────────────────────────────────────────────────────

/// The semantic type of a protocol message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageType {
    /// A request expecting a response.
    Request,
    /// A response to a prior [`MessageType::Request`].
    Response,
    /// An asynchronous notification of an occurrence.
    Event,
    /// A message addressed to all peers.
    Broadcast,
    /// Acknowledgement of receipt.
    Acknowledgment,
    /// Liveness probe.
    Heartbeat,
    /// An error condition.
    Error,
}

impl MessageType {
    fn as_u8(&self) -> u8 {
        match self {
            MessageType::Request => 0,
            MessageType::Response => 1,
            MessageType::Event => 2,
            MessageType::Broadcast => 3,
            MessageType::Acknowledgment => 4,
            MessageType::Heartbeat => 5,
            MessageType::Error => 6,
        }
    }

    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(MessageType::Request),
            1 => Some(MessageType::Response),
            2 => Some(MessageType::Event),
            3 => Some(MessageType::Broadcast),
            4 => Some(MessageType::Acknowledgment),
            5 => Some(MessageType::Heartbeat),
            6 => Some(MessageType::Error),
            _ => None,
        }
    }
}

// ── ProtocolVersion ───────────────────────────────────────────────────────────

/// Semantic version for the wire protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolVersion {
    /// Major version; must match for compatibility.
    pub major: u8,
    /// Minor version; backwards-compatible within the same major.
    pub minor: u8,
}

impl ProtocolVersion {
    /// Create a new protocol version.
    pub fn new(major: u8, minor: u8) -> Self {
        Self { major, minor }
    }

    /// Returns `true` if `self` is wire-compatible with `other`.
    ///
    /// Compatibility requires matching major versions.
    pub fn is_compatible(&self, other: &ProtocolVersion) -> bool {
        self.major == other.major
    }
}

impl fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

// ── ProtocolHeader ────────────────────────────────────────────────────────────

/// Fixed-layout header prepended to every protocol message.
#[derive(Debug, Clone)]
pub struct ProtocolHeader {
    /// Wire protocol version.
    pub version: ProtocolVersion,
    /// The semantic type of this message.
    pub message_type: MessageType,
    /// Monotonically increasing sequence number assigned by the sender.
    pub sequence_number: u64,
    /// Optional correlation ID linking request↔response pairs.
    pub correlation_id: Option<String>,
    /// Identifier of the sending agent.
    pub sender_id: String,
    /// Identifier of the intended recipient agent, or `None` for broadcasts.
    pub receiver_id: Option<String>,
    /// Creation time as Unix milliseconds.
    pub timestamp: u64,
    /// Maximum message lifetime in milliseconds; `None` means no expiry.
    pub ttl_ms: Option<u64>,
}

// ── ProtocolError ─────────────────────────────────────────────────────────────

/// Errors that can occur during message encoding, decoding, or validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    /// The encoded version is incompatible with the current implementation.
    InvalidVersion,
    /// The message checksum did not match the computed value.
    ChecksumMismatch,
    /// The message has exceeded its TTL.
    Expired,
    /// The payload bytes could not be interpreted.
    MalformedPayload,
    /// The sequence number was out of order.
    SequenceError,
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProtocolError::InvalidVersion => write!(f, "invalid protocol version"),
            ProtocolError::ChecksumMismatch => write!(f, "CRC-32 checksum mismatch"),
            ProtocolError::Expired => write!(f, "message TTL expired"),
            ProtocolError::MalformedPayload => write!(f, "malformed payload"),
            ProtocolError::SequenceError => write!(f, "sequence number out of order"),
        }
    }
}

// ── CRC-32 (polynomial 0xEDB88320) ───────────────────────────────────────────

/// Compute a CRC-32 checksum using the IEEE polynomial 0xEDB88320.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFF_FFFF
}

// ── Encoding helpers ──────────────────────────────────────────────────────────

fn encode_string(s: &str, buf: &mut Vec<u8>) {
    let bytes = s.as_bytes();
    let len = bytes.len() as u16;
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(bytes);
}

fn decode_string(buf: &[u8], pos: &mut usize) -> Option<String> {
    if *pos + 2 > buf.len() {
        return None;
    }
    let len = u16::from_be_bytes([buf[*pos], buf[*pos + 1]]) as usize;
    *pos += 2;
    if *pos + len > buf.len() {
        return None;
    }
    let s = String::from_utf8(buf[*pos..*pos + len].to_vec()).ok()?;
    *pos += len;
    Some(s)
}

fn encode_optional_string(opt: &Option<String>, buf: &mut Vec<u8>) {
    match opt {
        Some(s) => {
            buf.push(1);
            encode_string(s, buf);
        }
        None => buf.push(0),
    }
}

fn decode_optional_string(buf: &[u8], pos: &mut usize) -> Option<Option<String>> {
    if *pos >= buf.len() {
        return None;
    }
    let present = buf[*pos];
    *pos += 1;
    if present == 0 {
        Some(None)
    } else {
        Some(Some(decode_string(buf, pos)?))
    }
}

fn encode_optional_u64(opt: &Option<u64>, buf: &mut Vec<u8>) {
    match opt {
        Some(v) => {
            buf.push(1);
            buf.extend_from_slice(&v.to_be_bytes());
        }
        None => buf.push(0),
    }
}

fn decode_optional_u64(buf: &[u8], pos: &mut usize) -> Option<Option<u64>> {
    if *pos >= buf.len() {
        return None;
    }
    let present = buf[*pos];
    *pos += 1;
    if present == 0 {
        Some(None)
    } else {
        if *pos + 8 > buf.len() {
            return None;
        }
        let v = u64::from_be_bytes(buf[*pos..*pos + 8].try_into().ok()?);
        *pos += 8;
        Some(Some(v))
    }
}

// ── ProtocolMessage ───────────────────────────────────────────────────────────

/// A complete protocol message: header + payload + CRC-32 checksum.
#[derive(Debug, Clone)]
pub struct ProtocolMessage {
    /// Message header.
    pub header: ProtocolHeader,
    /// Arbitrary payload bytes.
    pub payload: Vec<u8>,
    /// CRC-32 checksum over `header_bytes || payload`.
    pub checksum: u32,
}

impl ProtocolMessage {
    /// Construct a new message, computing the checksum automatically.
    pub fn new(header: ProtocolHeader, payload: Vec<u8>) -> Self {
        let mut msg = Self {
            header,
            payload,
            checksum: 0,
        };
        msg.checksum = msg.compute_checksum();
        msg
    }

    /// Encode the message into a byte vector suitable for transmission.
    ///
    /// Layout:
    /// ```text
    /// [major: u8][minor: u8][msg_type: u8]
    /// [seq_num: u64 BE]
    /// [correlation_id: optional_string]
    /// [sender_id: string]
    /// [receiver_id: optional_string]
    /// [timestamp: u64 BE]
    /// [ttl_ms: optional_u64]
    /// [payload_len: u32 BE][payload: bytes]
    /// [checksum: u32 BE]
    /// ```
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Version.
        buf.push(self.header.version.major);
        buf.push(self.header.version.minor);

        // Message type.
        buf.push(self.header.message_type.as_u8());

        // Sequence number.
        buf.extend_from_slice(&self.header.sequence_number.to_be_bytes());

        // Correlation ID.
        encode_optional_string(&self.header.correlation_id, &mut buf);

        // Sender ID.
        encode_string(&self.header.sender_id, &mut buf);

        // Receiver ID.
        encode_optional_string(&self.header.receiver_id, &mut buf);

        // Timestamp.
        buf.extend_from_slice(&self.header.timestamp.to_be_bytes());

        // TTL.
        encode_optional_u64(&self.header.ttl_ms, &mut buf);

        // Payload.
        let payload_len = self.payload.len() as u32;
        buf.extend_from_slice(&payload_len.to_be_bytes());
        buf.extend_from_slice(&self.payload);

        // Checksum.
        buf.extend_from_slice(&self.checksum.to_be_bytes());

        buf
    }

    /// Decode a message from raw bytes.
    ///
    /// Returns [`ProtocolError::MalformedPayload`] if the bytes are truncated
    /// or contain invalid fields.  Checksum validity is verified during decode;
    /// use [`Self::verify_checksum`] to re-verify an already-decoded message.
    pub fn decode(bytes: &[u8]) -> Result<ProtocolMessage, ProtocolError> {
        let mut pos = 0;

        // Version.
        if pos + 2 > bytes.len() {
            return Err(ProtocolError::MalformedPayload);
        }
        let major = bytes[pos];
        let minor = bytes[pos + 1];
        pos += 2;
        // Accept any major version; callers may check compatibility separately.
        let version = ProtocolVersion { major, minor };

        // Message type.
        if pos >= bytes.len() {
            return Err(ProtocolError::MalformedPayload);
        }
        let message_type =
            MessageType::from_u8(bytes[pos]).ok_or(ProtocolError::MalformedPayload)?;
        pos += 1;

        // Sequence number.
        if pos + 8 > bytes.len() {
            return Err(ProtocolError::MalformedPayload);
        }
        let sequence_number = u64::from_be_bytes(
            bytes[pos..pos + 8]
                .try_into()
                .map_err(|_| ProtocolError::MalformedPayload)?,
        );
        pos += 8;

        // Correlation ID.
        let correlation_id = decode_optional_string(bytes, &mut pos)
            .ok_or(ProtocolError::MalformedPayload)?;

        // Sender ID.
        let sender_id =
            decode_string(bytes, &mut pos).ok_or(ProtocolError::MalformedPayload)?;

        // Receiver ID.
        let receiver_id = decode_optional_string(bytes, &mut pos)
            .ok_or(ProtocolError::MalformedPayload)?;

        // Timestamp.
        if pos + 8 > bytes.len() {
            return Err(ProtocolError::MalformedPayload);
        }
        let timestamp = u64::from_be_bytes(
            bytes[pos..pos + 8]
                .try_into()
                .map_err(|_| ProtocolError::MalformedPayload)?,
        );
        pos += 8;

        // TTL.
        let ttl_ms =
            decode_optional_u64(bytes, &mut pos).ok_or(ProtocolError::MalformedPayload)?;

        // Payload.
        if pos + 4 > bytes.len() {
            return Err(ProtocolError::MalformedPayload);
        }
        let payload_len = u32::from_be_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| ProtocolError::MalformedPayload)?,
        ) as usize;
        pos += 4;
        if pos + payload_len > bytes.len() {
            return Err(ProtocolError::MalformedPayload);
        }
        let payload = bytes[pos..pos + payload_len].to_vec();
        pos += payload_len;

        // Checksum (last 4 bytes).
        if pos + 4 > bytes.len() {
            return Err(ProtocolError::MalformedPayload);
        }
        let checksum = u32::from_be_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| ProtocolError::MalformedPayload)?,
        );

        let msg = ProtocolMessage {
            header: ProtocolHeader {
                version,
                message_type,
                sequence_number,
                correlation_id,
                sender_id,
                receiver_id,
                timestamp,
                ttl_ms,
            },
            payload,
            checksum,
        };

        if !msg.verify_checksum() {
            return Err(ProtocolError::ChecksumMismatch);
        }

        Ok(msg)
    }

    /// Compute the CRC-32 checksum over all header fields and the payload.
    fn compute_checksum(&self) -> u32 {
        // Encode everything except the checksum field itself.
        let mut data = Vec::new();
        data.push(self.header.version.major);
        data.push(self.header.version.minor);
        data.push(self.header.message_type.as_u8());
        data.extend_from_slice(&self.header.sequence_number.to_be_bytes());
        encode_optional_string(&self.header.correlation_id, &mut data);
        encode_string(&self.header.sender_id, &mut data);
        encode_optional_string(&self.header.receiver_id, &mut data);
        data.extend_from_slice(&self.header.timestamp.to_be_bytes());
        encode_optional_u64(&self.header.ttl_ms, &mut data);
        let payload_len = self.payload.len() as u32;
        data.extend_from_slice(&payload_len.to_be_bytes());
        data.extend_from_slice(&self.payload);
        crc32(&data)
    }

    /// Returns `true` if the stored checksum matches the computed checksum.
    pub fn verify_checksum(&self) -> bool {
        self.checksum == self.compute_checksum()
    }
}

// ── MessageSequencer ──────────────────────────────────────────────────────────

/// Monotonically increasing sequence number generator.
pub struct MessageSequencer {
    sequence: u64,
}

impl Default for MessageSequencer {
    fn default() -> Self {
        Self::new()
    }
}

impl MessageSequencer {
    /// Create a new sequencer starting at sequence number 0.
    pub fn new() -> Self {
        Self { sequence: 0 }
    }

    /// Return the next sequence number and advance the internal counter.
    pub fn next(&mut self) -> u64 {
        let seq = self.sequence;
        self.sequence += 1;
        seq
    }

    /// Returns `true` if `seq` equals `expected`.
    pub fn validate(seq: u64, expected: u64) -> bool {
        seq == expected
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn sample_header() -> ProtocolHeader {
        ProtocolHeader {
            version: ProtocolVersion::new(1, 0),
            message_type: MessageType::Request,
            sequence_number: 42,
            correlation_id: Some("corr-123".to_string()),
            sender_id: "agent-A".to_string(),
            receiver_id: Some("agent-B".to_string()),
            timestamp: 1_700_000_000_000,
            ttl_ms: Some(5_000),
        }
    }

    // ── ProtocolVersion ──

    #[test]
    fn version_compatibility_same_major() {
        let v1 = ProtocolVersion::new(1, 0);
        let v2 = ProtocolVersion::new(1, 5);
        assert!(v1.is_compatible(&v2));
    }

    #[test]
    fn version_incompatible_different_major() {
        let v1 = ProtocolVersion::new(1, 0);
        let v2 = ProtocolVersion::new(2, 0);
        assert!(!v1.is_compatible(&v2));
    }

    // ── CRC-32 ──

    #[test]
    fn crc32_empty() {
        // CRC-32 of empty bytes is 0x00000000 with this polynomial.
        assert_eq!(crc32(&[]), 0x0000_0000);
    }

    #[test]
    fn crc32_known_value() {
        // CRC-32 of b"123456789" = 0xCBF43926 per the standard.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn crc32_deterministic() {
        let a = crc32(b"hello world");
        let b = crc32(b"hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn crc32_different_inputs_differ() {
        assert_ne!(crc32(b"hello"), crc32(b"world"));
    }

    // ── Encode / Decode round-trip ──

    #[test]
    fn roundtrip_full_header() {
        let header = sample_header();
        let payload = b"Hello, World!".to_vec();
        let msg = ProtocolMessage::new(header, payload.clone());

        let encoded = msg.encode();
        let decoded = ProtocolMessage::decode(&encoded).unwrap();

        assert_eq!(decoded.header.version.major, 1);
        assert_eq!(decoded.header.version.minor, 0);
        assert_eq!(decoded.header.message_type, MessageType::Request);
        assert_eq!(decoded.header.sequence_number, 42);
        assert_eq!(
            decoded.header.correlation_id,
            Some("corr-123".to_string())
        );
        assert_eq!(decoded.header.sender_id, "agent-A");
        assert_eq!(decoded.header.receiver_id, Some("agent-B".to_string()));
        assert_eq!(decoded.header.timestamp, 1_700_000_000_000);
        assert_eq!(decoded.header.ttl_ms, Some(5_000));
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn roundtrip_no_optional_fields() {
        let header = ProtocolHeader {
            version: ProtocolVersion::new(2, 1),
            message_type: MessageType::Heartbeat,
            sequence_number: 0,
            correlation_id: None,
            sender_id: "ping-agent".to_string(),
            receiver_id: None,
            timestamp: 0,
            ttl_ms: None,
        };
        let msg = ProtocolMessage::new(header, vec![]);
        let decoded = ProtocolMessage::decode(&msg.encode()).unwrap();
        assert_eq!(decoded.header.correlation_id, None);
        assert_eq!(decoded.header.receiver_id, None);
        assert_eq!(decoded.header.ttl_ms, None);
        assert!(decoded.payload.is_empty());
    }

    #[test]
    fn roundtrip_all_message_types() {
        let types = [
            MessageType::Request,
            MessageType::Response,
            MessageType::Event,
            MessageType::Broadcast,
            MessageType::Acknowledgment,
            MessageType::Heartbeat,
            MessageType::Error,
        ];
        for mt in types {
            let header = ProtocolHeader {
                version: ProtocolVersion::new(1, 0),
                message_type: mt.clone(),
                sequence_number: 1,
                correlation_id: None,
                sender_id: "a".to_string(),
                receiver_id: None,
                timestamp: 0,
                ttl_ms: None,
            };
            let msg = ProtocolMessage::new(header, vec![0x42]);
            let decoded = ProtocolMessage::decode(&msg.encode()).unwrap();
            assert_eq!(decoded.header.message_type, mt);
        }
    }

    #[test]
    fn verify_checksum_passes() {
        let msg = ProtocolMessage::new(sample_header(), b"data".to_vec());
        assert!(msg.verify_checksum());
    }

    #[test]
    fn verify_checksum_fails_on_tamper() {
        let msg = ProtocolMessage::new(sample_header(), b"data".to_vec());
        let mut encoded = msg.encode();
        // Flip a byte in the payload area.
        let last = encoded.len() - 5;
        encoded[last] ^= 0xFF;
        let result = ProtocolMessage::decode(&encoded);
        assert_eq!(result.unwrap_err(), ProtocolError::ChecksumMismatch);
    }

    #[test]
    fn decode_truncated_returns_error() {
        let msg = ProtocolMessage::new(sample_header(), b"data".to_vec());
        let encoded = msg.encode();
        let truncated = &encoded[..5];
        assert_eq!(
            ProtocolMessage::decode(truncated).unwrap_err(),
            ProtocolError::MalformedPayload
        );
    }

    // ── MessageSequencer ──

    #[test]
    fn sequencer_monotone() {
        let mut seq = MessageSequencer::new();
        assert_eq!(seq.next(), 0);
        assert_eq!(seq.next(), 1);
        assert_eq!(seq.next(), 2);
    }

    #[test]
    fn sequencer_validate_correct() {
        assert!(MessageSequencer::validate(5, 5));
    }

    #[test]
    fn sequencer_validate_incorrect() {
        assert!(!MessageSequencer::validate(5, 6));
    }

    // ── Unicode payload ──

    #[test]
    fn roundtrip_unicode_payload() {
        let header = ProtocolHeader {
            version: ProtocolVersion::new(1, 0),
            message_type: MessageType::Event,
            sequence_number: 99,
            correlation_id: Some("こんにちは".to_string()),
            sender_id: "宇宙人".to_string(),
            receiver_id: Some("地球人".to_string()),
            timestamp: 12345,
            ttl_ms: None,
        };
        let payload = "🦀 Rust is great! 🚀".as_bytes().to_vec();
        let msg = ProtocolMessage::new(header, payload.clone());
        let decoded = ProtocolMessage::decode(&msg.encode()).unwrap();
        assert_eq!(decoded.payload, payload);
        assert_eq!(decoded.header.sender_id, "宇宙人");
    }
}
