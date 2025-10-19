// mobile/DeteccionHumo.js - Versión básica para Expo
import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Alert } from 'react-native';
import { Camera, useCameraDevices } from 'expo-camera';

export default function DeteccionHumo() {
  const [permission, setPermission] = useState(null);
  const devices = useCameraDevices('back');
  const device = devices?.back;

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setPermission(status === 'granted');
    })();
  }, []);

  if (permission === null) return <Text>Solicitando permiso...</Text>;
  if (permission === false) return <Text>Sin permiso.</Text>;

  return (
    <View style={styles.container}>
      {device && (
        <Camera style={styles.camera} device={device}>
          <View style={styles.overlay}>
            <Text style={styles.text}>Apunta la cámara a humo para detectar</Text>
            {/* Aquí integra ONNX en useEffect para procesar frames */}
          </View>
        </Camera>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  camera: { flex: 1 },
  overlay: { position: 'absolute', bottom: 50, alignSelf: 'center' },
  text: { color: 'white', fontSize: 16 },
});